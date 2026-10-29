import pytest

from biomaterial_optimizer.data.materiom import parse_methods, parse_recipe


def test_recipe_row(materiom_recipe):
    recipe, _, _ = parse_recipe(materiom_recipe)
    assert recipe == {
        "recipe_id": "abc123",
        "name": "Eggshell bio-ceramic",
        "source": "10.1000/example.1",
    }


def test_source_falls_back_to_group(materiom_recipe):
    materiom_recipe["recipe"]["data"]["details"] = {}
    recipe, _, _ = parse_recipe(materiom_recipe)
    assert recipe["source"] == "materiom:group:g1"


def test_ingredients(materiom_recipe):
    _, ingredients, _ = parse_recipe(materiom_recipe)
    assert [(i["ingredient"], i["amount"], i["unit"]) for i in ingredients] == [
        ("Eggshell powder", 60.0, "g"),
        ("Sodium Alginate", 5.0, "g"),
    ]


def test_properties_keep_reported_text(materiom_recipe):
    _, _, properties = parse_recipe(materiom_recipe)
    by_name = {p["property"]: p for p in properties}
    assert set(by_name) == {
        "Flexural Modulus",
        "Tensile Strength",
        "Elongation at Break (%)",
        "UV transmittance",
    }
    assert (
        by_name["Tensile Strength"]["value"],
        by_name["Tensile Strength"]["sd"],
    ) == (43.9, 2.3)
    assert by_name["UV transmittance"]["value"] is None
    assert by_name["UV transmittance"]["reported"] == "<0.1"


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Poured into plates and dried at 45 C for 12 h.",
            {"drying temperature": 45.0, "drying time": 12.0},
        ),
        (
            (
                "Cast and dried at room temperature (22 ± 2 °C) for 48 h. The dried film "
                "was peeled off and conditioned at 25 °C."
            ),
            {"drying temperature": 22.0, "drying time": 48.0},
        ),
        (
            "Cast on glass and dried at normal ambient temperature for 3 days.",
            {"drying temperature": 25.0, "drying time": 72.0},
        ),
        (
            # Raw seaweed drying comes first; the film's drying step is the last one.
            (
                "The seaweed was ovendried at 40 ◦C for 72 h. The solution was poured "
                "into a tray and dried in an oven at 50 ◦C for 24 h."
            ),
            {"drying temperature": 50.0, "drying time": 24.0},
        ),
        (
            (
                "Dried in an oven at 60 °C for 4 h. Then films were immersed in "
                "cross-linking solutions (1.5% CaCl2) for 5 min, washed and dried at "
                "30 °C for 1 h."
            ),
            {"drying temperature": 60.0, "drying time": 4.0, "crosslinking time": 5.0},
        ),
        (
            "The film was dried at 40 ℃ until constant weight.",
            {"drying temperature": 40.0},
        ),
        ("Films were immersed in sunflower oil for 10 s.", {}),  # not a crosslinker
        ("Samples were oven-dried for about 24–36 h.", {}),  # no temperature, a range
    ],
)
def test_parse_methods(text, expected):
    assert parse_methods(text) == pytest.approx(expected)


def test_processing_becomes_ingredient_rows(materiom_recipe):
    materiom_recipe["recipe"]["data"]["methods"] = [
        {"type": "other", "description": "Cast and dried at 50 °C for 24 h."}
    ]
    _, ingredients, _ = parse_recipe(materiom_recipe)
    rows = {i["ingredient"]: (i["amount"], i["unit"]) for i in ingredients}
    assert rows["drying temperature"] == (50.0, "°C")
    assert rows["drying time"] == (24.0, "h")
