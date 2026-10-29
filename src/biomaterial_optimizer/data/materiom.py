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


# "dried at/in/for ...", not "dried film" or "after drying,".
_DRY = re.compile(
    r"dr(?:ied|ying)(?= (?:at|in|for|under|on|using|it)\b| ?\()", re.IGNORECASE
)
_CROSSLINK_STEP = re.compile(r"immers|dipp|soak", re.IGNORECASE)
_CROSSLINKER = re.compile(r"cacl|calcium|ca2\+|chloride|cross-?link", re.IGNORECASE)
_TEMPERATURE = re.compile(
    r"(?<![\d.–-])(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?\s*)?(?:°|º|◦|\uf0b0|o)?\s*(?:C\b|℃)"
)
_DURATION = re.compile(
    r"(?<![\d.–-])(?:±\s*)?(\d+(?:\.\d+)?)\s*(h|hours?|hrs?|min|minutes?|s|sec|seconds?|days?)\b",
    re.IGNORECASE,
)
_HOURS = {"h": 1, "hour": 1, "hr": 1, "min": 1 / 60, "minute": 1 / 60, "day": 24}
_HOURS |= {"s": 1 / 3600, "sec": 1 / 3600, "second": 1 / 3600}


def _hours(match: re.Match) -> float:
    unit = match[2].lower()
    return float(match[1]) * _HOURS[unit if unit in _HOURS else unit.rstrip("s")]


def _window(text: str, start: int, size: int = 130) -> str:
    """Text from `start` to the end of the sentence, at most `size` characters."""
    end = text.find(". ", start)
    return text[start : min(end if end != -1 else len(text), start + size)]


def parse_methods(text: str) -> dict[str, float]:
    """Drying temperature (°C), drying time (h) and crosslinking time (min) from methods text.

    Only what the text states plainly is returned; anything unclear is left out.
    ASSUMPTIONS: the film's drying step is the last "dried at ..." with a temperature
    before the crosslinking bath (earlier ones are usually drying the raw seaweed;
    later ones re-dry after the bath); "room"/"ambient" without a number means 25 °C.
    """
    text = re.sub(r"\s+", " ", text)
    found = {}
    bath_start = len(text)
    for m in _CROSSLINK_STEP.finditer(text):
        window = _window(text, m.start(), 200)
        if _CROSSLINKER.search(window) and (d := _DURATION.search(window)):
            found["crosslinking time"] = _hours(d) * 60
            bath_start = m.start()
            break
    for m in _DRY.finditer(text, 0, bath_start):
        window = _window(text, m.start())
        if t := _TEMPERATURE.search(window):
            found["drying temperature"] = float(t[1])
        elif re.search(r"room|ambient", window, re.IGNORECASE):
            found["drying temperature"] = 25.0
        else:
            continue
        found.pop("drying time", None)
        if d := _DURATION.search(window):
            found["drying time"] = _hours(d)
    return found


PROCESSING_UNITS = {
    "drying temperature": "°C",
    "drying time": "h",
    "crosslinking time": "min",
}


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

    # Processing steps are stored like treatments: as ingredient rows.
    methods = " ".join(
        m.get("description") or ""
        for m in data.get("methods") or []
        if isinstance(m, dict)
    )
    for name, amount in parse_methods(methods).items():
        ingredients.append(
            {
                "recipe_id": recipe_id,
                "ingredient": name,
                "amount": round(amount, 4),
                "unit": PROCESSING_UNITS[name],
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
