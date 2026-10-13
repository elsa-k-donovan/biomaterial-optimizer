# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "duckdb"]
# ///
"""Fetch Materiom Commons recipes into DuckDB, with ingredient amounts and measured properties.

Stage 1: list every recipe ID from the site's Algolia search index.
Stage 2: fetch each recipe's full record from the Materiom backend (1 request/second),
         caching each one as JSON so a rerun skips recipes already downloaded.
Stage 3: build DuckDB tables: recipes, ingredients, properties, plus a view of
         recipes that have at least one measured property value.


Usage (from the repo root):
    uv run --env-file .env scripts/fetch_materiom.py --limit 10      # trial run
    uv run --env-file .env scripts/fetch_materiom.py                 # everything (~11 min)
    uv run --env-file .env scripts/fetch_materiom.py --inspect <recipe_id>
    uv run scripts/fetch_materiom.py --rebuild-only                  # rebuild DB from cache

Data: Materiom Commons. Recipes carry their own licence (stored per recipe);
the platform terms specify CC BY-NC-SA 4.0. Keep attribution and recipe IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import httpx

DETAIL_URL = "https://commons-backend.materiom.org/api/materials-database/recipes/{id}"
RECIPE_PAGE_URL = (
    "https://commons.materiom.org/recipe/{id}"  # for attribution; check on site
)
REQUEST_PAUSE_S = 1.0  # be polite: at most one request per second
MAX_RETRIES = 4
USER_AGENT = "biomaterial-optimizer research script (personal, non-commercial)"

EMPTY_STRINGS = {"", "-", "n/a", "na", "null", "none", "nan"}


# ================================================================ helpers


def is_filled(value: Any) -> bool:
    """True if a value holds real data. Numbers (including 0) count; blank strings,
    placeholders like 'N/A', None and empty containers do not."""
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return value.strip().lower() not in EMPTY_STRINGS
    if isinstance(value, dict):
        return any(is_filled(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(is_filled(v) for v in value)
    return True


def to_number(value: Any) -> float | None:
    """Parse '12.5', '12,5' or 12.5 to a float; None for ranges, text or blanks."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            return None
    return None


_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
_MEAN_SD = re.compile(
    rf"^\s*(?P<q>[~≈<>≤≥]?)\s*(?P<mean>{_NUM})\s*(?:±|\+/-|\+-)\s*(?P<sd>{_NUM})"
)
_RANGE = re.compile(rf"^\s*(?P<lo>{_NUM})\s*(?:-|–|—|to)\s*(?P<hi>{_NUM})\s*[^\d]*$")
_SINGLE = re.compile(rf"^\s*(?P<q>[~≈<>≤≥]?)\s*(?P<num>{_NUM})\s*[^\d]*$")


def _parse_float(text: str) -> float:
    # '12,5' is a decimal comma; '1,477' (three digits after) is a thousands separator.
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", text):
        return float(text.replace(",", ""))
    return float(text.replace(",", "."))


def parse_measurement(value: Any) -> dict[str, Any]:
    """Parse a measured value into mean, standard deviation, range and qualifier.

    Handles 12.5, '12.5', '12,5', '43.9 ± 2.3', '43.9 +/- 2.3', '20-30', '~15', '>50',
    '12.5 MPa'. 'value' is the mean, or the midpoint of a range. Anything else
    (free text) gives value None, and the raw text is kept elsewhere.
    """
    out: dict[str, Any] = {
        "value": None,
        "sd": None,
        "min": None,
        "max": None,
        "qualifier": None,
    }
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        out["value"] = to_number(value)
        return out
    if not isinstance(value, str):
        return out
    text = value.strip()
    if m := _MEAN_SD.match(text):
        out["value"] = _parse_float(m["mean"])
        out["sd"] = abs(_parse_float(m["sd"]))
        out["qualifier"] = m["q"] or None
    elif m := _RANGE.match(text):
        lo, hi = sorted((_parse_float(m["lo"]), _parse_float(m["hi"])))
        out.update(value=(lo + hi) / 2, min=lo, max=hi, qualifier="range")
    elif m := _SINGLE.match(text):
        out["value"] = _parse_float(m["num"])
        out["qualifier"] = m["q"] or None
    return out


_UNIT_ALIASES = {
    "mpa": "MPa",
    "gpa": "GPa",
    "kpa": "kPa",
    "pa": "Pa",
    "deg": "deg",
    "°": "deg",
    "degree": "deg",
    "degrees": "deg",
    "%": "%",
    "g/cm3": "g/cm3",
    "g/cm³": "g/cm3",
    "kg/m3": "kg/m3",
    "kg/m³": "kg/m3",
    "µm": "µm",
    "um": "µm",
    "μm": "µm",
    "°c": "°C",
    "c": "°C",
}


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return _UNIT_ALIASES.get(unit.strip().lower(), unit.strip())


def normalize_property(name: str | None) -> tuple[str | None, str | None]:
    """'Elongation at Break (%)' -> ('elongation at break', '%');
    "Young's Modulus" -> 'youngs modulus'. Returns (name, unit found in brackets)."""
    if not name:
        return None, None
    text = name.strip().lower().replace("’", "'")
    bracket_unit = None
    if m := re.search(r"\(([^)]*)\)\s*$", text):
        inner = m[1].strip()
        if normalize_unit(inner) in _UNIT_ALIASES.values():
            bracket_unit = normalize_unit(inner)
            text = text[: m.start()]
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9%() ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip() or None, bracket_unit


# Names that refer to the same ingredient. Kept short and conservative on purpose;
# extend it as you find more duplicates in the data.
_INGREDIENT_ALIASES = {
    "sodium alginate": "alginate",
    "glycerin": "glycerol",
    "glycerine": "glycerol",
    "distilled water": "water",
    "deionized water": "water",
    "deionised water": "water",
    "cacl2": "calcium chloride",
    "cacl2 2h2o": "calcium chloride",
    "calcium chloride dihydrate": "calcium chloride",
}


def normalize_ingredient(name: str | None) -> str | None:
    if not name:
        return None
    text = re.sub(r"[^a-z0-9 ]+", " ", name.strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    return _INGREDIENT_ALIASES.get(text, text) or None


def normalize_source(ref: str | None) -> str | None:
    """Reduce a paper link to a stable ID: the DOI if one is present
    ('https://pubs.acs.org/doi/10.1021/acsabm.0c01353' -> '10.1021/acsabm.0c01353'),
    otherwise the link without protocol, 'www.' or trailing slash."""
    if not ref:
        return None
    text = ref.strip()
    if m := re.search(r"10\.\d{4,9}/[^\s?#]+", text):
        return m[0].rstrip("/.").lower()
    text = re.sub(r"^https?://(www\.)?", "", text, flags=re.IGNORECASE)
    return text.rstrip("/").lower() or None


def as_text(value: Any) -> str | None:
    """Flatten a value that may be a string, a number, or a {value, label} dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("label", "value", "name", "amount"):
            if is_filled(value.get(key)):
                return as_text(value[key])
        return None
    text = str(value).strip()
    return text if is_filled(text) else None


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and is_filled(item[key]):
            return item[key]
    return None


def measured_value(item: dict[str, Any]) -> Any:
    """A property's value may be a plain value or a dict like {'amount': '12'}."""
    raw = item.get("value")
    if isinstance(raw, dict):
        return first_present(raw, ("amount", "value", "mean", "avg", "min", "max"))
    return raw


# ================================================================ stage 1: list IDs


def get_algolia_config() -> tuple[str, str, str]:
    names = [
        "MATERIOM_ALGOLIA_APP_ID",
        "MATERIOM_ALGOLIA_API_KEY",
        "MATERIOM_ALGOLIA_INDEX",
    ]
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(
            f"Missing environment variables: {', '.join(missing)}. "
            "Run from the repo root with: uv run --env-file .env ..."
        )
    return tuple(os.environ[n] for n in names)  # type: ignore[return-value]


def request_with_retry(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            wait = 2**attempt
            print(f"  network error ({exc.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2**attempt
            print(f"  HTTP {resp.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        return resp
    raise RuntimeError(f"Gave up after {MAX_RETRIES} attempts: {url}")


def list_recipe_ids(cache_dir: Path) -> list[dict[str, Any]]:
    """Return the search-index summary for every recipe (incl. objectID)."""
    app_id, api_key, index = get_algolia_config()
    url = f"https://{app_id}-dsn.algolia.net/1/indexes/{index}/query"
    headers = {"X-Algolia-Application-Id": app_id, "X-Algolia-API-Key": api_key}

    summaries: list[dict[str, Any]] = []
    page = 0
    with httpx.Client(headers=headers, timeout=30) as client:
        while True:
            resp = request_with_retry(
                client,
                "POST",
                url,
                json={"query": "", "page": page, "hitsPerPage": 100},
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])
            for h in hits:
                h.pop("_highlightResult", None)
            summaries.extend(hits)
            page += 1
            if page >= data.get("nbPages", 0) or not hits:
                break
            time.sleep(REQUEST_PAUSE_S)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "search_index.json").write_text(json.dumps(summaries, indent=2))
    print(f"Stage 1: found {len(summaries)} recipes in the search index.")
    return summaries


# ================================================================ stage 2: details


def fetch_details(ids: list[str], detail_dir: Path) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (detail_dir / f"{i}.json").exists()]
    print(
        f"Stage 2: {len(ids) - len(todo)} already cached, {len(todo)} to fetch "
        f"(~{len(todo) * REQUEST_PAUSE_S / 60:.0f} min)."
    )

    failed: list[str] = []
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        for n, recipe_id in enumerate(todo, 1):
            resp = request_with_retry(client, "GET", DETAIL_URL.format(id=recipe_id))
            if resp.status_code == 200:
                (detail_dir / f"{recipe_id}.json").write_text(resp.text)
            else:
                failed.append(recipe_id)
                print(f"  {recipe_id}: HTTP {resp.status_code}, skipped")
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} fetched")
            time.sleep(REQUEST_PAUSE_S)

    if failed:
        print(f"  {len(failed)} recipes failed; rerun the script to retry them.")


# ================================================================ stage 3: parse


def parse_recipe(detail: dict[str, Any]) -> tuple[dict, list[dict], list[dict]]:
    recipe = detail.get("recipe") or {}
    data = recipe.get("data") or {}
    recipe_id = str(recipe.get("_id") or data.get("unique_id") or "")
    author = detail.get("author") or {}
    details = data.get("details") or {}

    ingredients = []
    for pos, ing in enumerate(data.get("ingredients") or []):
        amount_raw = (
            (ing.get("value") or {}).get("amount")
            if isinstance(ing.get("value"), dict)
            else ing.get("value")
        )
        unit = ing.get("unit") or {}
        ingredients.append(
            {
                "recipe_id": recipe_id,
                "position": pos,
                "ingredient": as_text(ing.get("label")),
                "ingredient_norm": normalize_ingredient(as_text(ing.get("label"))),
                "amount_raw": as_text(amount_raw),
                "amount": to_number(amount_raw),
                "unit": (
                    as_text(unit.get("value"))
                    if isinstance(unit, dict)
                    else as_text(unit)
                ),
                "unit_label": (
                    as_text(unit.get("label")) if isinstance(unit, dict) else None
                ),
            }
        )

    properties = []
    for section_key, section in data.items():
        if not section_key.endswith("Properties") or not isinstance(section, dict):
            continue
        for pos, prop in enumerate(section.get("properties") or []):
            if not isinstance(prop, dict):
                continue
            name = as_text(first_present(prop, ("name", "label", "property", "type")))
            method = as_text(first_present(prop, ("method", "standard", "testMethod")))

            # Real measurements live in prop["items"]: [{"value": 1477, "unit": "MPa"}, ...].
            # The site also stores empty template slots for every property; those have
            # no filled items and are skipped.
            if "items" in prop:
                measurements = [
                    m for m in (prop.get("items") or []) if isinstance(m, dict)
                ]
            elif to_number(measured_value(prop)) is not None or is_filled(
                prop.get("unit")
            ):
                measurements = [
                    prop
                ]  # fallback: a value stored directly on the property
            else:
                measurements = []

            for idx, m in enumerate(measurements):
                value = measured_value(m)
                if not is_filled(value):
                    continue  # a named property with no measurement isn't data
                parsed = parse_measurement(value)
                prop_norm, bracket_unit = normalize_property(name)
                unit = as_text(m.get("unit")) or as_text(prop.get("unit"))
                properties.append(
                    {
                        "recipe_id": recipe_id,
                        "section": section_key,
                        "position": pos,
                        "item_index": idx,
                        "property": name,
                        "property_norm": prop_norm,
                        "value_raw": as_text(value),
                        "value": parsed["value"],
                        "value_sd": parsed["sd"],
                        "value_min": parsed["min"],
                        "value_max": parsed["max"],
                        "qualifier": parsed["qualifier"],
                        "unit": unit,
                        "unit_norm": normalize_unit(unit) or bracket_unit,
                        "method": as_text(
                            first_present(m, ("method", "standard", "testMethod"))
                        )
                        or method,
                        "item_json": json.dumps(m),
                    }
                )

    paper = None
    for src in details.get("sources") or []:
        if isinstance(src, dict):
            paper = as_text(src.get("source")) or as_text(src.get("name"))
        elif isinstance(src, str):
            paper = as_text(src)
        if paper:
            break

    row = {
        "recipe_id": recipe_id,
        "name": as_text(data.get("name")),
        "author": " ".join(
            filter(
                None, [as_text(author.get("name")), as_text(author.get("last_name"))]
            )
        )
        or None,
        "licence": as_text(details.get("license") or details.get("licence")),
        "import_type": as_text(detail.get("importType")),
        # Source for leave-one-source-out validation: the cited paper (DOI) if there is
        # one, else the import group (usually one paper), else the recipe itself.
        "source_id": normalize_source(paper)
        or as_text(data.get("group_id"))
        or recipe_id,
        "source_paper": paper,
        "group_id": as_text(data.get("group_id")),
        "n_ingredients": len(ingredients),
        "n_ingredients_with_amount": sum(i["amount"] is not None for i in ingredients),
        "n_properties": len(properties),
        "n_numeric_properties": sum(p["value"] is not None for p in properties),
        "source_url": RECIPE_PAGE_URL.format(id=recipe_id),
        "created_at": as_text(recipe.get("created_at")),
        "raw_json": json.dumps(detail),
    }
    return row, ingredients, properties


def build_database(detail_dir: Path, db_path: Path) -> None:
    files = sorted(detail_dir.glob("*.json"))
    if not files:
        sys.exit(
            f"No cached recipes in {detail_dir}. Run without --rebuild-only first."
        )

    recipes, ingredients, properties = [], [], []
    for f in files:
        try:
            r, ings, props = parse_recipe(json.loads(f.read_text()))
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"  could not parse {f.name}: {exc}")
            continue
        if not r["recipe_id"]:
            r["recipe_id"] = f.stem
        recipes.append(r)
        ingredients.extend(ings)
        properties.extend(props)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE OR REPLACE TABLE recipes (
            recipe_id VARCHAR PRIMARY KEY, name VARCHAR, author VARCHAR, licence VARCHAR,
            import_type VARCHAR, source_id VARCHAR, source_paper VARCHAR, group_id VARCHAR,
            n_ingredients INTEGER, n_ingredients_with_amount INTEGER,
            n_properties INTEGER, n_numeric_properties INTEGER, source_url VARCHAR,
            created_at VARCHAR, raw_json JSON, fetched_at TIMESTAMPTZ)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ingredients (
            recipe_id VARCHAR, position INTEGER, ingredient VARCHAR, ingredient_norm VARCHAR,
            amount_raw VARCHAR,
            amount DOUBLE, unit VARCHAR, unit_label VARCHAR)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE properties (
            recipe_id VARCHAR, section VARCHAR, position INTEGER, item_index INTEGER,
            property VARCHAR, property_norm VARCHAR, value_raw VARCHAR, value DOUBLE,
            value_sd DOUBLE, value_min DOUBLE, value_max DOUBLE, qualifier VARCHAR,
            unit VARCHAR, unit_norm VARCHAR, method VARCHAR, item_json JSON)
    """)

    now = datetime.now(timezone.utc).isoformat()
    rcols = [
        "recipe_id",
        "name",
        "author",
        "licence",
        "import_type",
        "source_id",
        "source_paper",
        "group_id",
        "n_ingredients",
        "n_ingredients_with_amount",
        "n_properties",
        "n_numeric_properties",
        "source_url",
        "created_at",
        "raw_json",
    ]
    con.executemany(
        f"INSERT INTO recipes VALUES ({', '.join('?' * (len(rcols) + 1))})",
        [[r[c] for c in rcols] + [now] for r in recipes],
    )
    icols = [
        "recipe_id",
        "position",
        "ingredient",
        "ingredient_norm",
        "amount_raw",
        "amount",
        "unit",
        "unit_label",
    ]
    if ingredients:
        con.executemany(
            f"INSERT INTO ingredients VALUES ({', '.join('?' * len(icols))})",
            [[i[c] for c in icols] for i in ingredients],
        )
    pcols = [
        "recipe_id",
        "section",
        "position",
        "item_index",
        "property",
        "property_norm",
        "value_raw",
        "value",
        "value_sd",
        "value_min",
        "value_max",
        "qualifier",
        "unit",
        "unit_norm",
        "method",
        "item_json",
    ]
    if properties:
        con.executemany(
            f"INSERT INTO properties VALUES ({', '.join('?' * len(pcols))})",
            [[p[c] for c in pcols] for p in properties],
        )

    con.execute("""
        CREATE OR REPLACE VIEW recipes_with_properties AS
        SELECT * EXCLUDE (raw_json) FROM recipes WHERE n_properties > 0
    """)
    con.execute("""
        CREATE OR REPLACE VIEW property_coverage AS
        SELECT property_norm AS property, unit_norm AS unit,
               count(DISTINCT recipe_id) AS n_recipes,
               count(DISTINCT recipe_id) FILTER (WHERE value IS NOT NULL) AS n_numeric,
               count(DISTINCT recipe_id) FILTER (WHERE value_sd IS NOT NULL) AS n_with_sd,
               string_agg(DISTINCT property, ' | ') AS original_names
        FROM properties GROUP BY ALL ORDER BY n_numeric DESC, n_recipes DESC
    """)

    total, with_props, numeric = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE n_properties > 0),
               count(*) FILTER (WHERE n_numeric_properties > 0) FROM recipes
    """).fetchone()
    print(
        f"Stage 3: {total} recipes in {db_path}; {with_props} have measured properties "
        f"({numeric} with at least one numeric value)."
    )
    print("\nMost common properties:")
    print(con.sql("SELECT * FROM property_coverage LIMIT 15"))
    con.close()


# ================================================================ inspect


def inspect_detail(recipe_id: str) -> None:
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        resp = request_with_retry(client, "GET", DETAIL_URL.format(id=recipe_id))
    resp.raise_for_status()
    detail = resp.json()
    row, ings, props = parse_recipe(detail)
    print(
        f"{row['name']}  ({row['n_ingredients']} ingredients, {row['n_properties']} properties)\n"
    )
    for i in ings:
        print(f"  ingredient: {i['ingredient']}: {i['amount_raw']} {i['unit'] or ''}")
    for p in props:
        print(
            f"  property:   [{p['section']}] {p['property']}: {p['value_raw']} {p['unit'] or ''}"
        )
    data = (detail.get("recipe") or {}).get("data") or {}
    for key, section in data.items():
        if (
            key.endswith("Properties")
            and isinstance(section, dict)
            and section.get("properties")
        ):
            print(f"\nRaw items in {key}:")
            print(json.dumps(section["properties"], indent=2)[:3000])


# ================================================================ main


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--inspect",
        metavar="RECIPE_ID",
        help="fetch one recipe, show what the parser extracts, and exit",
    )
    p.add_argument(
        "--limit", type=int, help="only fetch the first N recipes (trial run)"
    )
    p.add_argument(
        "--rebuild-only",
        action="store_true",
        help="skip downloading; rebuild the database from cached JSON",
    )
    p.add_argument("--db", type=Path, default=Path("data/materiom.duckdb"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/raw/materiom"))
    args = p.parse_args()

    if args.inspect:
        inspect_detail(args.inspect)
        return

    detail_dir = args.cache_dir / "recipes"
    if not args.rebuild_only:
        summaries = list_recipe_ids(args.cache_dir)
        ids = [str(s["objectID"]) for s in summaries if s.get("objectID")]
        if args.limit:
            ids = ids[: args.limit]
        fetch_details(ids, detail_dir)
    build_database(detail_dir, args.db)


if __name__ == "__main__":
    main()
