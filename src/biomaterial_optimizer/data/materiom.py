"""Turn one downloaded Materiom recipe (raw JSON) into recipe, ingredient and property rows.

Rows are plain dicts with the same fields as the models in `schema.py`;
`build.py` cleans the names and validates them.
"""

import re

from .schema import parse_reported

_DOI = re.compile(r"10\.\d{4,9}/[^\s?#]+")


def _text(value) -> str | None:
    """A value as clean text, or None. Dicts like {'value': 'g', 'label': 'grams'} give 'value'."""
    if isinstance(value, dict):
        value = (
            value.get("value") if value.get("value") is not None else value.get("label")
        )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source(data: dict, recipe_id: str) -> str:
    """The cited paper's DOI if there is one, else the Materiom import group, else the recipe."""
    for source in (data.get("details") or {}).get("sources") or []:
        ref = (
            _text(source.get("source") or source.get("name"))
            if isinstance(source, dict)
            else _text(source)
        )
        if ref:
            doi = _DOI.search(ref)
            return doi[0].rstrip("/.").lower() if doi else ref
    group = _text(data.get("group_id"))
    return f"materiom:group:{group}" if group else f"materiom:recipe:{recipe_id}"


def parse_recipe(raw: dict) -> tuple[dict, list[dict], list[dict]]:
    """Return (recipe, ingredients, properties) rows for one Materiom recipe."""
    data = (raw.get("recipe") or {}).get("data") or {}
    recipe_id = str(raw["recipe"]["_id"])
    recipe = {
        "recipe_id": recipe_id,
        "name": _text(data.get("name")),
        "source": _source(data, recipe_id),
    }

    ingredients = []
    for item in data.get("ingredients") or []:
        name = _text(item.get("label"))
        if not name:
            continue
        value = item.get("value")
        amount_text = _text(value.get("amount") if isinstance(value, dict) else value)
        amount = parse_reported(amount_text)[0] if amount_text else None
        ingredients.append(
            {
                "recipe_id": recipe_id,
                "ingredient": name,
                "amount": amount,
                "unit": _text(item.get("unit")),
            }
        )

    properties = []
    for key, section in data.items():
        if not key.endswith("Properties") or not isinstance(section, dict):
            continue
        for prop in section.get("properties") or []:
            name = _text(prop.get("name"))
            # Measurements live in "items"; empty template slots have none.
            for item in prop.get("items") or []:
                reported = _text(item.get("value"))
                if not name or not reported:
                    continue
                value, sd = parse_reported(reported)
                properties.append(
                    {
                        "recipe_id": recipe_id,
                        "property": name,
                        "value": value,
                        "sd": sd,
                        "unit": _text(item.get("unit")),
                        "reported": reported,
                    }
                )
    return recipe, ingredients, properties
