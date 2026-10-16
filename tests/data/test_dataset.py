import json

import duckdb
import pytest

from biomaterial_optimizer.data.build import build_database
from biomaterial_optimizer.data.dataset import (
    Skip,
    build_dataset,
    per_gram_base,
    recipe_inputs,
)

ROLES = {
    "alginate": ("base", None),
    "glycerol": ("plasticiser", 1.26),
    "water": ("solvent", 1.0),
    "calcium chloride bath": ("crosslink_bath", None),
    "beeswax coating": ("surface_treatment", None),
}


@pytest.mark.parametrize(
    "amount, unit, density, base_amount, base_unit, expected",
    [
        (0.25, "g", None, 1.0, "g", 0.25),  # grams / grams
        (250, "mg", None, 1.0, "g", 0.25),
        (2.0, "mL", 1.26, 10.0, "g", 0.252),  # volume needs a density
        (30.0, "% w/w", None, 2.0, "% w/v", 0.30),  # already per 100 g of base
        (1.0, "% w/v", None, 2.0, "% w/v", 0.5),  # two solution concentrations
        (10.0, "g/L", None, 2.0, "% w/v", 0.5),  # 2 % w/v = 20 g/L
        (40.0, "%", None, 60.0, "%", 40 / 60),  # parts of the same mixture
        (5.0, "%", None, 5.0, "g", 0.05),  # bare % read as % of base weight
    ],
)
def test_per_gram_base(amount, unit, density, base_amount, base_unit, expected):
    assert per_gram_base(
        amount, unit, density, base_amount, base_unit
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "unit, density, base_unit",
    [
        ("g", None, "% w/v"),  # mass vs concentration
        ("mL", None, "g"),  # volume without a density
        ("M", None, "g"),  # molar needs a molar mass
    ],
)
def test_per_gram_base_refuses_to_guess(unit, density, base_unit):
    with pytest.raises(Skip):
        per_gram_base(1.0, unit, density, 1.0, base_unit)


def item(ingredient, amount, unit):
    return {"ingredient": ingredient, "amount": amount, "unit": unit}


def test_recipe_inputs():
    row = recipe_inputs(
        [
            item("alginate", 10.0, "g"),
            item("glycerol", 3.0, "g"),
            item("water", 987.0, "g"),
            item("calcium chloride bath", 2.0, "% w/v"),
            item("calcium chloride bath", 10.0, "g/L"),  # = 1 % w/v
            item("beeswax coating", 2.0, "coats"),
            item("unknown extract", 0.1, "g"),
        ],
        ROLES,
    )
    assert row["base"] == "alginate"
    assert row["plasticiser"] == pytest.approx(0.3)
    assert row["crosslink_bath"] == pytest.approx(3.0)
    assert row["surface_treatment"] == 2.0
    assert row["other"] == pytest.approx(0.01)  # no role -> other
    assert row["filler"] == 0.0


@pytest.mark.parametrize(
    "items, reason",
    [
        ([item("alginate", 0.0, "g")], "base amount is 0"),
        ([item("alginate", None, "g")], "base polymer has no amount"),
        ([item("alginate", 1.0, "g"), item("glycerol", None, "g")], "has no amount"),
        (
            [item("alginate", 1.0, "g"), item("calcium chloride bath", 2.0, "g")],
            "bath not a concentration",
        ),
    ],
)
def test_recipe_inputs_skips(items, reason):
    with pytest.raises(Skip, match=reason):
        recipe_inputs(items, ROLES)


def test_build_dataset(tmp_path, materiom_recipe, literature_dir, aliases_path):
    materiom_dir = tmp_path / "materiom"
    materiom_dir.mkdir()
    (materiom_dir / "abc123.json").write_text(json.dumps(materiom_recipe))
    db = tmp_path / "test.duckdb"
    build_database(db, materiom_dir, literature_dir, aliases_path)

    out = tmp_path / "processed" / "table.parquet"
    roles = literature_dir.parent / "ingredient_roles.csv"
    n, skipped, _ = build_dataset(db, roles, out)

    con = duckdb.connect()
    assert con.execute(f"SELECT count(*) FROM '{out}'").fetchone() == (n,)
    row = con.execute(
        f"SELECT base, plasticiser, filler FROM '{out}' WHERE recipe_id = 'abdollahi2013-CN5'"
    ).fetchone()
    assert row == ("alginate", pytest.approx(0.25), pytest.approx(0.05))
    assert (out.parent / "skipped.csv").exists()
    assert all(s["reason"] for s in skipped)
