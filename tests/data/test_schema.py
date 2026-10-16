import duckdb
import pytest
from pydantic import ValidationError

from biomaterial_optimizer.data.schema import (
    DUCKDB_TABLES,
    Ingredient,
    Property,
    Recipe,
    parse_reported,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("43.9", (43.9, None)),
        ("43.9 ± 2.3", (43.9, 2.3)),
        ("12,5", (12.5, None)),  # decimal comma
        ("1,477", (1477.0, None)),  # thousands separator
        ("<0.1", (None, None)),  # a limit: kept as text only
        ("20-30", (None, None)),  # a range: kept as text only
        ("off-white", (None, None)),
    ],
)
def test_parse_reported(text, expected):
    assert parse_reported(text) == expected


def test_recipe_needs_source():
    with pytest.raises(ValidationError):
        Recipe(recipe_id="r1", source="")


def test_ingredient_amount_needs_unit():
    Ingredient(recipe_id="r1", ingredient="glycerol", amount=5, unit="g")
    with pytest.raises(ValidationError):
        Ingredient(recipe_id="r1", ingredient="glycerol", amount=5, unit="")


def test_property_keeps_text_without_value():
    p = Property(recipe_id="r1", property="uv transmittance", unit="%", reported="<0.1")
    assert p.value is None and p.reported == "<0.1"


@pytest.mark.parametrize(
    "fields",
    [
        {"value": "117.7", "reported": "117.7"},  # value without unit
        {"sd": "1", "unit": "deg", "reported": "± 1"},  # sd without value
        {"value": "1", "sd": "-1", "unit": "deg", "reported": "1"},  # negative sd
        {"value": "1", "unit": "deg", "reported": ""},  # nothing reported
    ],
)
def test_property_rejects_invalid(fields):
    with pytest.raises(ValidationError):
        Property(recipe_id="r1", property="water contact angle", **fields)


def test_duckdb_tables_create():
    con = duckdb.connect()
    con.execute(DUCKDB_TABLES)
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert tables == {"recipes", "ingredients", "properties"}
