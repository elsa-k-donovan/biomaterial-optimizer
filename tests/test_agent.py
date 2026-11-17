"""The agent's tools and wiring, offline: a fake model stands in for Claude."""

import json

import pytest
from pydantic_ai.models.test import TestModel

from biomaterial_optimizer.agent import Deps, agent, coverage, run_query
from biomaterial_optimizer.data.build import build_database
from biomaterial_optimizer.data.dataset import build_dataset


@pytest.fixture
def deps(tmp_path, materiom_recipe, literature_dir, aliases_path):
    materiom_dir = tmp_path / "materiom"
    materiom_dir.mkdir()
    (materiom_dir / "abc123.json").write_text(json.dumps(materiom_recipe))
    db = tmp_path / "test.duckdb"
    build_database(db, materiom_dir, literature_dir, aliases_path)
    table = tmp_path / "processed" / "table.parquet"
    build_dataset(db, literature_dir.parent / "ingredient_roles.csv", table)
    return Deps(db_path=db, table_path=table)


def test_query_reads(deps):
    out = run_query(deps.db_path, "SELECT count(*) AS n FROM recipes")
    assert "n" in out


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM recipes",
        "CREATE TABLE t AS SELECT 1",
        "COPY recipes TO 'leak.csv'",
        "SELECT * FROM read_csv('data/aliases.yaml')",
    ],
)
def test_query_cannot_write_or_touch_files(deps, sql):
    assert run_query(deps.db_path, sql).startswith("Query failed")


def test_query_errors_come_back_as_text(deps):
    assert "Query failed" in run_query(deps.db_path, "SELEC nonsense")


def test_coverage(deps):
    out = coverage(deps.table_path, "water_contact_angle")
    assert set(out) == {"absolute", "change_vs_control"}
    assert out["absolute"]["papers"] >= out["change_vs_control"]["papers"]


def test_agent_calls_every_tool(deps):
    # TestModel calls each tool once with valid arguments, then answers.
    with agent.override(model=TestModel()):
        result = agent.run_sync("What can I try?", deps=deps)
    output = json.loads(result.output)
    assert set(output) == {"query_database", "data_coverage", "propose_recipes"}
    assert isinstance(output["propose_recipes"], list)
