"""Build the model table: one row per seaweed recipe, ready for ML.

Inputs (what a scientist controls):
  base               main seaweed polymer: alginate, agar, carrageenan or seaweed
  plasticiser ...    grams per gram of base polymer, summed per role
                     (roles come from data/ingredient_roles.csv)
  crosslink_bath     % w/v of the crosslinking bath the film is dipped in
  surface_treatment  number of coatings or surface treatments

Targets: one main measurement per property (see TARGETS), each with its sd.

A recipe is a "seaweed recipe" when one of its ingredients has the role `base`.
Recipes that can't be converted go to skipped.csv next to the table, with the reason.
"""

import csv
from pathlib import Path

import duckdb

ADDITIVE_ROLES = [
    "plasticiser",
    "copolymer",
    "filler",
    "crosslinker",
    "water_repellent",
    "active",
    "other",
]

# Column name -> (property in the database, its unit).
TARGETS = {
    "water_contact_angle": ("water contact angle", "deg"),
    "water_solubility": ("water solubility", "%"),
    "uv_transmittance_280nm": ("uv transmittance 280 nm", "%"),
    "antioxidant_dpph": ("antioxidant activity dpph", "%"),
    "tensile_strength": ("tensile strength", "MPa"),
    "elongation_at_break": ("elongation at break", "%"),
}

BASE_TYPES = ["alginate", "agar", "carrageenan", "seaweed"]

# Sources left out of the table, with the reason (it goes to skipped.csv).
EXCLUDED_SOURCES = {
    # Every recipe lists 500 g cellulose per 4 g seaweed, identical across recipes
    # with different results: the amount that varies was lost when entered in Materiom.
    "10.15376/biores.14.2.3389-3410": "cellulose amounts look wrongly entered",
}

GRAMS = {"g": 1.0, "mg": 1e-3, "kg": 1e3}
MILLILITRES = {
    "mL": 1.0,
    "L": 1e3,
    "tsp": 4.93,
    "tbsp": 14.79,
    "Cups": 240.0,
    "drops": 0.05,
}
GRAMS_PER_LITRE = {"% w/v": 10.0, "g/L": 1.0, "mg/mL": 1.0}
# Units that already mean "per 100 g of base polymer".
PER_100G_BASE = {"% w/w", "phr", "% w/w macroalgae", "% gN/gA"}

COLUMNS = (
    [
        "recipe_id",
        "source",
        "base",
        *ADDITIVE_ROLES,
        "crosslink_bath",
        "surface_treatment",
    ]
    + [c for t in TARGETS for c in (t, f"{t}_sd")]
    + ["ingredients"]
)


class Skip(Exception):
    """A recipe that can't go in the table; the message is the reason."""


def _common(amount: float, unit: str | None, density: float | None):
    """('mass', grams) or ('conc', g per litre of solution), or None if not convertible."""
    if unit in GRAMS:
        return "mass", amount * GRAMS[unit]
    if unit in MILLILITRES and density:
        return "mass", amount * MILLILITRES[unit] * density
    if unit in GRAMS_PER_LITRE:
        return "conc", amount * GRAMS_PER_LITRE[unit]
    if unit == "% v/v" and density:
        return "conc", amount * 10 * density
    return None


def per_gram_base(
    amount: float,
    unit: str | None,
    density: float | None,
    base_amount: float,
    base_unit: str | None,
) -> float:
    """Grams of an ingredient per gram of base polymer. Raises Skip if it can't tell."""
    if unit in PER_100G_BASE:
        return amount / 100
    if unit == base_unit == "%":  # both are parts of the same mixture
        return amount / base_amount
    if unit == "%":
        # ASSUMPTION: a bare "%" on an additive (base not in "%") means % of the
        # base polymer's weight, the usual convention in film papers, e.g.
        # "seaweed 5 g | lignin 5 %" -> 0.05 g lignin per g seaweed.
        return amount / 100
    a, b = _common(amount, unit, density), _common(base_amount, base_unit, None)
    if a and b and a[0] == b[0]:
        return a[1] / b[1]
    raise Skip(f"can't compare {unit} with base in {base_unit}")


def recipe_inputs(ingredients: list[dict], roles: dict) -> dict:
    """Input columns for one recipe. `ingredients` rows have ingredient, amount, unit."""
    bases = [
        i for i in ingredients if roles.get(i["ingredient"], ("other",))[0] == "base"
    ]
    if any(b["amount"] is None for b in bases):
        raise Skip("base polymer has no amount")
    if len({b["unit"] for b in bases}) > 1:
        raise Skip("base polymers in different units")
    base_amount, base_unit = sum(b["amount"] for b in bases), bases[0]["unit"]
    if base_amount <= 0:
        raise Skip("base amount is 0")

    main = max(bases, key=lambda b: b["amount"])["ingredient"]
    amounts = {role: 0.0 for role in ADDITIVE_ROLES}
    amounts |= {"crosslink_bath": 0.0, "surface_treatment": 0.0}
    for i in ingredients:
        name, amount, unit = i["ingredient"], i["amount"], i["unit"]
        role, density = roles.get(name, ("other", None))
        if role in ("base", "solvent"):
            continue
        if role == "surface_treatment":
            amounts[role] += amount if unit == "coats" and amount is not None else 1
            continue
        if amount is None:
            raise Skip(f"{name} has no amount")
        if role == "crosslink_bath":
            # ASSUMPTION: a bare "%" for a bath means % w/v (g per 100 mL of bath).
            per_100ml = {"% w/v": 1.0, "%": 1.0, "g/L": 0.1, "mg/mL": 0.1}
            if unit not in per_100ml:
                raise Skip(f"{name} bath not a concentration")
            amounts[role] += amount * per_100ml[unit]
            continue
        amounts[role] += per_gram_base(amount, unit, density, base_amount, base_unit)
    return {"base": next(t for t in BASE_TYPES if t in main), **amounts}


def read_roles(path: Path) -> dict[str, tuple[str, float | None]]:
    with path.open(newline="") as f:
        return {
            r["ingredient"]: (
                r["role"],
                float(r["density_g_ml"]) if r["density_g_ml"] else None,
            )
            for r in csv.DictReader(f)
        }


def build_dataset(
    db_path: Path, roles_path: Path, out_path: Path
) -> tuple[int, list[dict], set[str]]:
    """Write the model table. Returns (rows written, skipped recipes, ingredients without a role)."""
    roles = read_roles(roles_path)
    con = duckdb.connect(str(db_path), read_only=True)
    sources = dict(con.execute("SELECT recipe_id, source FROM recipes").fetchall())
    ingredients: dict[str, list[dict]] = {}
    for recipe_id, name, amount, unit in con.execute(
        "SELECT recipe_id, ingredient, amount, unit FROM ingredients"
    ).fetchall():
        ingredients.setdefault(recipe_id, []).append(
            {"ingredient": name, "amount": amount, "unit": unit}
        )
    measured: dict[str, dict] = {}
    for column, (prop, unit) in TARGETS.items():
        for recipe_id, value, sd in con.execute(
            "SELECT recipe_id, value, sd FROM properties"
            " WHERE property = ? AND unit = ? AND value IS NOT NULL",
            [prop, unit],
        ).fetchall():
            measured.setdefault(recipe_id, {}).update(
                {column: value, f"{column}_sd": sd}
            )
    con.close()

    rows, skipped, unlisted = [], [], set()
    for recipe_id, items in sorted(ingredients.items()):
        if not any(roles.get(i["ingredient"], ("other",))[0] == "base" for i in items):
            continue  # not seaweed-based
        try:
            if sources[recipe_id] in EXCLUDED_SOURCES:
                raise Skip(EXCLUDED_SOURCES[sources[recipe_id]])
            if recipe_id not in measured:
                raise Skip("no target measured")
            inputs = recipe_inputs(items, roles)
        except Skip as e:
            skipped.append({"recipe_id": recipe_id, "reason": str(e)})
            continue
        unlisted |= {i["ingredient"] for i in items} - roles.keys()
        names = sorted(
            {
                i["ingredient"]
                for i in items
                if roles.get(i["ingredient"], ("",))[0] != "solvent"
            }
        )
        row = {
            "recipe_id": recipe_id,
            "source": sources[recipe_id],
            "ingredients": names,
        }
        rows.append(row | inputs | measured[recipe_id])

    _write(out_path, rows)
    with (out_path.parent / "skipped.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["recipe_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped)
    return len(rows), skipped, unlisted


def _write(out_path: Path, rows: list[dict]) -> None:
    def sql_type(column):
        if column == "ingredients":
            return "VARCHAR[]"
        return "VARCHAR" if column in ("recipe_id", "source", "base") else "DOUBLE"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t ({', '.join(f'{c} {sql_type(c)}' for c in COLUMNS)})")
    if rows:
        con.executemany(
            f"INSERT INTO t VALUES ({', '.join('?' * len(COLUMNS))})",
            [[row.get(c) for c in COLUMNS] for row in rows],
        )
    con.execute("COPY t TO ? (FORMAT parquet)", [str(out_path)])
    con.close()
