"""The eval's grader and runner, offline: a perfect answer must pass, an empty one fail."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from biomaterial_optimizer.agent import Deps, agent
from biomaterial_optimizer.agent_eval import grade, load_cases, mentions_number, run_all
from biomaterial_optimizer.data.build import build_database

CASES = load_cases(Path(__file__).parents[1] / "evals" / "agent_cases.yaml")


@pytest.fixture
def db(tmp_path, materiom_recipe, literature_dir, aliases_path):
    materiom_dir = tmp_path / "materiom"
    materiom_dir.mkdir()
    (materiom_dir / "abc123.json").write_text(json.dumps(materiom_recipe))
    path = tmp_path / "test.duckdb"
    build_database(path, materiom_dir, literature_dir, aliases_path)
    return path


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "text, value, expected",
    [
        ("about 22 MPa", 22.4, True),
        ("1,200 recipes", 1200, True),
        ("15 MPa", 22.4, False),
    ],
)
def test_mentions_number(text, value, expected):
    assert mentions_number(text, value) is expected


def test_perfect_answer_passes(db):
    c = case("one-recipe-value")
    call = ToolCallPart("query_database", {"sql": "SELECT ..."})
    checks = grade(c, "CN5 has a tensile strength of 22.4 MPa.", [call], db)
    assert all(checks.values()), checks


def test_empty_answer_fails(db):
    checks = grade(case("one-recipe-value"), "", [], db)
    assert not any(checks.values())


def test_tool_arguments_use_defaults(db):
    c = case("propose-alginate")  # wants base=alginate, which is the default
    call = ToolCallPart("propose_recipes", {})
    answer = "Test these in the lab; the bath was never tried for water resistance."
    assert all(grade(c, answer, [call], db).values())
    wrong = ToolCallPart("propose_recipes", {"base": "agar"})
    assert not grade(c, answer, [wrong], db)["propose_recipes arguments"]


def test_runner_writes_rows_and_records_errors(db, tmp_path):
    def answer(messages, info):
        if "explode" in str(messages[-1]):
            raise RuntimeError("API down")
        return ModelResponse(parts=[TextPart("There is too little data to say.")])

    cases = [case("coverage-microbes"), {"id": "boom", "question": "explode"}]
    with agent.override(model=FunctionModel(answer)):
        rows, errors = asyncio.run(run_all(cases, Deps(db_path=db), tmp_path / "run"))
    assert [r["prompt_id"] for r in rows] == ["coverage-microbes"]
    assert rows[0]["grade"] is True
    assert [e["prompt_id"] for e in errors] == ["boom"]
    lines = (tmp_path / "run" / "results.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["transcript"]  # full conversation saved
