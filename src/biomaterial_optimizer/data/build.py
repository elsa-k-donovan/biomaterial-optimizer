"""Build the database: Materiom JSON + literature CSVs -> three DuckDB tables.

Steps, in order:
  1. read     Materiom files and literature CSVs into plain rows
  2. clean    names and units, using data/aliases.yaml
  3. check    every row against the models in schema.py
  4. write    the valid rows to recipes / ingredients / properties

Rows that fail a check are not dropped silently: they are written to a
`skipped.csv` next to the database, with the reason.
"""

import csv
import json
from pathlib import Path

import duckdb
import yaml
from pydantic import ValidationError

from .materiom import parse_recipe
from .schema import DUCKDB_TABLES, Ingredient, Property, Recipe

MODELS = {"recipes": Recipe, "ingredients": Ingredient, "properties": Property}


# ---- 1. read -------------------------------------------------------------------


def read_materiom(folder: Path) -> dict[str, list[dict]]:
    rows = {"recipes": [], "ingredients": [], "properties": []}
    for path in sorted(folder.glob("*.json")):
        recipe, ingredients, properties = parse_recipe(json.loads(path.read_text()))
        rows["recipes"].append(recipe)
        rows["ingredients"] += ingredients
        rows["properties"] += properties
    return rows


def read_literature(folder: Path) -> dict[str, list[dict]]:
    rows = {}
    for table in MODELS:
        with (folder / f"{table}.csv").open(newline="") as f:
            rows[table] = list(csv.DictReader(f))
    return rows


# ---- 2. clean ------------------------------------------------------------------


def _key(name: str) -> str:
    return " ".join(str(name).lower().split())


def clean(rows: dict[str, list[dict]], aliases: dict) -> None:
    """Fix names and units in place: lowercase, then apply aliases.yaml."""
    ingredient_names = aliases.get("ingredients") or {}
    property_names = aliases.get("properties") or {}
    units = {_key(k): v for k, v in (aliases.get("units") or {}).items()}
    default_units = aliases.get("default_units") or {}

    def unit(value):
        return (
            units.get(_key(value), value.strip())
            if value and str(value).strip()
            else None
        )

    for row in rows["ingredients"]:
        name = _key(row["ingredient"])
        row["ingredient"] = ingredient_names.get(name, name)
        row["unit"] = unit(row.get("unit"))
    for row in rows["properties"]:
        name = _key(row["property"])
        row["property"] = property_names.get(name, name)
        row["unit"] = unit(row.get("unit")) or default_units.get(row["property"])


# ---- 3. check ------------------------------------------------------------------


def skipped_row(table: str, row: dict, reason: str) -> dict:
    """One line of skipped.csv."""
    return {
        "table": table,
        "recipe_id": row.get("recipe_id"),
        "reason": reason,
        "row": json.dumps(row),
    }


def check(rows: dict[str, list[dict]]) -> tuple[dict[str, list], list[dict]]:
    """Validate rows; return (valid models per table, skipped rows with reasons)."""
    valid, skipped = {table: [] for table in MODELS}, []

    def skip(table, row, reason):
        skipped.append(skipped_row(table, row, reason))

    seen = set()
    for row in rows["recipes"]:
        if row["recipe_id"] in seen:
            skip("recipes", row, "duplicate recipe_id")
            continue
        try:
            valid["recipes"].append(Recipe(**row))
            seen.add(row["recipe_id"])
        except ValidationError as e:
            skip("recipes", row, e.errors()[0]["msg"])

    for table in ("ingredients", "properties"):
        for row in rows[table]:
            if row.get("recipe_id") not in seen:
                skip(table, row, "unknown recipe_id")
                continue
            try:
                valid[table].append(MODELS[table](**row))
            except ValidationError as e:
                skip(table, row, e.errors()[0]["msg"])
    return valid, skipped


# ---- 4. write ------------------------------------------------------------------


def write(db_path: Path, valid: dict[str, list]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # always rebuilt from scratch
    con = duckdb.connect(str(db_path))
    con.execute(DUCKDB_TABLES)
    for table, models in valid.items():  # recipes first, so references resolve
        if models:
            fields = list(MODELS[table].model_fields)
            con.executemany(
                f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                [[getattr(m, f) for f in fields] for m in models],
            )
    con.close()


# ---- all together --------------------------------------------------------------


def build_database(
    db_path: Path, materiom_dir: Path, literature_dir: Path, aliases_path: Path
) -> tuple[dict[str, int], list[dict]]:
    """Run all four steps. Returns (row counts per table, skipped rows)."""
    materiom = (
        read_materiom(materiom_dir)
        if materiom_dir.exists()
        else {t: [] for t in MODELS}
    )
    literature = read_literature(literature_dir)

    # A paper already imported by Materiom must not be counted twice.
    in_materiom = {r["source"].lower() for r in materiom["recipes"]}
    duplicate = {
        r["recipe_id"]
        for r in literature["recipes"]
        if r["source"].lower() in in_materiom
    }
    skipped_dupes = [
        skipped_row(t, r, "paper already in Materiom")
        for t in MODELS
        for r in literature[t]
        if r["recipe_id"] in duplicate
    ]

    rows = {
        t: materiom[t] + [r for r in literature[t] if r["recipe_id"] not in duplicate]
        for t in MODELS
    }
    clean(rows, yaml.safe_load(aliases_path.read_text()) or {})
    valid, skipped = check(rows)
    write(db_path, valid)

    # Always rewritten, so a clean build never leaves an old skipped.csv behind.
    skipped = skipped_dupes + skipped
    with (db_path.parent / "skipped.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["table", "recipe_id", "reason", "row"])
        writer.writeheader()
        writer.writerows(skipped)
    return {t: len(v) for t, v in valid.items()}, skipped
