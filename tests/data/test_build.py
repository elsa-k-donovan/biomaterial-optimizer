import json

import duckdb

from biomaterial_optimizer.data.build import build_database


def build(tmp_path, materiom_recipe, literature_dir, aliases_path):
    materiom_dir = tmp_path / "materiom"
    materiom_dir.mkdir()
    (materiom_dir / "abc123.json").write_text(json.dumps(materiom_recipe))
    db = tmp_path / "test.duckdb"
    counts, skipped = build_database(db, materiom_dir, literature_dir, aliases_path)
    return duckdb.connect(str(db)), counts, skipped


def test_builds_three_tables(tmp_path, materiom_recipe, literature_dir, aliases_path):
    con, counts, _ = build(tmp_path, materiom_recipe, literature_dir, aliases_path)
    assert counts["recipes"] > 50  # 1 from Materiom + the literature
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert tables == {"recipes", "ingredients", "properties"}


def test_names_and_units_are_cleaned(
    tmp_path, materiom_recipe, literature_dir, aliases_path
):
    con, _, _ = build(tmp_path, materiom_recipe, literature_dir, aliases_path)
    ingredients = {
        r[0]
        for r in con.execute(
            "SELECT ingredient FROM ingredients WHERE recipe_id='abc123'"
        ).fetchall()
    }
    assert ingredients == {"eggshell powder", "alginate"}
    props = dict(
        con.execute(
            "SELECT property, unit FROM properties WHERE recipe_id='abc123'"
        ).fetchall()
    )
    assert props["flexural modulus"] == "MPa"  # 'Mpa' fixed
    assert props["elongation at break"] == "%"  # name fixed, blank unit filled


def test_old_skipped_csv_is_replaced(
    tmp_path, materiom_recipe, literature_dir, aliases_path
):
    (tmp_path / "skipped.csv").write_text("old run\n")
    build(tmp_path, materiom_recipe, literature_dir, aliases_path)
    text = (tmp_path / "skipped.csv").read_text()
    assert text.startswith("table,recipe_id,reason,row")
    assert "old run" not in text


def test_paper_already_in_materiom_is_not_counted_twice(
    tmp_path, materiom_recipe, literature_dir, aliases_path
):
    materiom_recipe["recipe"]["data"]["details"] = {
        "sources": [{"source": "10.3390/polym14194147"}]
    }  # Surya 2022
    con, _, skipped = build(tmp_path, materiom_recipe, literature_dir, aliases_path)
    surya = con.execute(
        "SELECT recipe_id FROM recipes WHERE recipe_id LIKE 'surya2022-%'"
    ).fetchall()
    assert surya == []
    assert any(s["reason"] == "paper already in Materiom" for s in skipped)
