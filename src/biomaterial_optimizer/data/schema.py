"""Data models: every biomaterial is a recipe, its ingredients and its measured properties.

Materiom and the literature both end up in these three shapes, and in the
three DuckDB tables with the same names.
"""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Row(BaseModel):
    """Shared settings: trim whitespace, reject unknown fields, treat blank CSV cells as missing."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_missing(cls, value):
        return None if isinstance(value, str) and value.strip() == "" else value


class Recipe(_Row):
    recipe_id: str = Field(min_length=1)
    name: str | None = None
    source: str = Field(min_length=1, description="DOI or Materiom URL")


class Ingredient(_Row):
    recipe_id: str = Field(min_length=1)
    ingredient: str = Field(min_length=1)
    amount: float | None = Field(default=None, ge=0)
    unit: str | None = None

    @model_validator(mode="after")
    def _amount_needs_unit(self):
        if self.amount is not None and self.unit is None:
            raise ValueError(f"{self.ingredient}: amount without a unit")
        return self


class Property(_Row):
    recipe_id: str = Field(min_length=1)
    property: str = Field(min_length=1)
    value: float | None = None  # only when the source reported a plain number
    sd: float | None = Field(default=None, ge=0)
    unit: str | None = None
    reported: str = Field(min_length=1, description="exactly what the source said")

    @model_validator(mode="after")
    def _check(self):
        if self.value is not None and self.unit is None:
            raise ValueError(f"{self.property}: value without a unit")
        if self.sd is not None and self.value is None:
            raise ValueError(f"{self.property}: sd without a value")
        return self


_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PLAIN = re.compile(rf"^({_NUMBER})$")
_WITH_SD = re.compile(rf"^({_NUMBER})\s*(?:±|\+/-)\s*({_NUMBER})$")
_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+$")  # '1,477' means 1477, not 1.477


def parse_reported(text: str) -> tuple[float | None, float | None]:
    """Return (value, sd) for '43.9' or '43.9 ± 2.3'; (None, None) for anything else.

    Limits ('<0.1') and ranges ('20-30') are deliberately not converted:
    they are kept as text in `reported`, never turned into a single number.
    """
    text = text.strip()
    text = text.replace(",", "") if _THOUSANDS.match(text) else text.replace(",", ".")
    if m := _PLAIN.match(text):
        return float(m[1]), None
    if m := _WITH_SD.match(text):
        return float(m[1]), abs(float(m[2]))
    return None, None


# The same three shapes as DuckDB tables.
DUCKDB_TABLES = """
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id  VARCHAR PRIMARY KEY,
    name       VARCHAR,
    source     VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS ingredients (
    recipe_id  VARCHAR NOT NULL REFERENCES recipes (recipe_id),
    ingredient VARCHAR NOT NULL,
    amount     DOUBLE,
    unit       VARCHAR
);
CREATE TABLE IF NOT EXISTS properties (
    recipe_id  VARCHAR NOT NULL REFERENCES recipes (recipe_id),
    property   VARCHAR NOT NULL,
    value      DOUBLE,
    sd         DOUBLE,
    unit       VARCHAR,
    reported   VARCHAR NOT NULL
);
"""
