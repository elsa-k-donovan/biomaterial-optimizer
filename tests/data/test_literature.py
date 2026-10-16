"""Checks on the hand-extracted literature data in data/literature/."""

import pytest
import yaml

from biomaterial_optimizer.data.build import MODELS, read_literature


@pytest.fixture
def literature(literature_dir):
    return read_literature(literature_dir)


@pytest.mark.parametrize("table", MODELS)
def test_every_row_is_valid(literature, table):
    for row in literature[table]:
        MODELS[table](**row)  # raises if invalid


def test_recipe_ids_are_unique(literature):
    ids = [r["recipe_id"] for r in literature["recipes"]]
    assert len(ids) == len(set(ids))


def test_every_row_belongs_to_a_recipe(literature):
    ids = {r["recipe_id"] for r in literature["recipes"]}
    for table in ("ingredients", "properties"):
        orphans = {r["recipe_id"] for r in literature[table]} - ids
        assert not orphans, f"{table}: unknown recipe ids {orphans}"


def test_every_source_is_documented(literature_dir, literature):
    documented = {
        entry["doi"]
        for entry in yaml.safe_load(
            (literature_dir / "sources.yaml").read_text()
        ).values()
    }
    used = {r["source"] for r in literature["recipes"]}
    assert used <= documented, f"missing from sources.yaml: {used - documented}"
