from biomaterial_optimizer.data.materiom import parse_recipe


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
