"""Evaluate the agent: ask it the questions in evals/agent_cases.yaml and grade each
answer with code (no human, no second LLM).

An eval is a test suite for an AI system. Normal tests check exact outputs; an LLM's
answers are worded differently every time, so instead each case checks *properties*
of the answer: did it call the right tool with the right arguments, does it contain
the right number (taken from the database), does it warn about thin data.
A case passes when all its checks pass; the headline score is the share of cases
that pass.
"""

import asyncio
import inspect
import json
import re
import time
from pathlib import Path

import duckdb
import yaml
from pydantic.fields import FieldInfo
from pydantic_ai.messages import ToolCallPart

from . import agent as agent_module
from .agent import Deps, agent


def load_cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())


def tool_calls(messages) -> list[ToolCallPart]:
    return [
        part
        for message in messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolCallPart)
    ]


def _defaults(tool: str) -> dict:
    """A tool's default arguments, so an argument the agent left out still counts."""
    params = inspect.signature(getattr(agent_module, tool)).parameters.values()
    return {
        p.name: p.default.default if isinstance(p.default, FieldInfo) else p.default
        for p in params
        if p.default is not inspect.Parameter.empty
    }


def _same(got, want) -> bool:
    if isinstance(want, int | float) and isinstance(got, int | float):
        return abs(got - want) < 1e-6
    return got == want


def mentions_number(text: str, value: float) -> bool:
    """True if `text` contains `value`, allowing rounding (22.4 matches "22", "22.40")."""
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]
    return any(abs(n - value) <= 0.01 * abs(value) + 0.51 for n in numbers)


def grade(
    case: dict,
    answer: str,
    calls: list[ToolCallPart],
    db_path: Path,
    recipes_before: int | None = None,
) -> dict[str, bool]:
    """Each check in the case -> passed or not."""
    checks = {}
    names = [c.tool_name for c in calls]
    for tool in case.get("tools", []):
        checks[f"called {tool}"] = tool in names
    for tool in case.get("no_tools", []):
        checks[f"avoided {tool}"] = tool not in names
    for tool, want in case.get("args", {}).items():
        defaults = _defaults(tool)
        checks[f"{tool} arguments"] = any(
            all(_same((defaults | c.args_as_dict()).get(k), v) for k, v in want.items())
            for c in calls
            if c.tool_name == tool
        )
    if sql := case.get("facts_sql"):
        con = duckdb.connect(str(db_path), read_only=True)
        (value,) = con.sql(sql).fetchone() or (None,)
        con.close()
        checks["right number"] = value is not None and mentions_number(answer, value)
    for words in case.get("mentions", []):
        checks[f"mentions {words[0]!r}"] = any(
            w.lower() in answer.lower() for w in words
        )
    if case.get("recipes_kept"):
        con = duckdb.connect(str(db_path), read_only=True)
        (after,) = con.sql("SELECT count(*) FROM recipes").fetchone() or (0,)
        con.close()
        checks["recipes kept"] = after == recipes_before
    return checks


async def run_case(case: dict, deps: Deps, timeout_s: float) -> dict:
    """Ask one question; return its result row (grade, answer, usage, transcript)."""
    con = duckdb.connect(str(deps.db_path), read_only=True)
    (recipes_before,) = con.sql("SELECT count(*) FROM recipes").fetchone() or (0,)
    con.close()
    start = time.monotonic()
    result = await asyncio.wait_for(
        agent.run(case["question"], deps=deps), timeout=timeout_s
    )
    calls = tool_calls(result.all_messages())
    checks = grade(case, result.output, calls, deps.db_path, recipes_before)
    usage = result.usage()
    return {
        "prompt_id": case["id"],
        "prompt": case["question"],
        "grade": all(checks.values()),
        "checks": checks,
        "answer": result.output,
        "tool_calls": [{"name": c.tool_name, "args": c.args_as_dict()} for c in calls],
        "model": result.response.model_name,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
        "latency_s": round(time.monotonic() - start, 1),
        "transcript": json.loads(result.all_messages_json()),
    }


async def run_all(
    cases: list[dict],
    deps: Deps,
    out_dir: Path,
    concurrency: int = 4,
    timeout_s: float = 300,
) -> tuple[list[dict], list[dict]]:
    """Run every case (a few at a time); write each row as soon as it finishes.

    Failures that aren't the agent's fault (API errors, timeouts) go to errors.jsonl
    instead of counting as a wrong answer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(concurrency)
    rows, errors = [], []

    async def one(case):
        async with limit:
            try:
                row = await run_case(case, deps, timeout_s)
            # Recorded, not raised: one failed case shouldn't stop the whole run.
            except Exception as e:  # noqa: BLE001
                kind = "timeout" if isinstance(e, TimeoutError) else type(e).__name__
                error = {"prompt_id": case["id"], "error": kind, "detail": str(e)[:500]}
                errors.append(error)
                with (out_dir / "errors.jsonl").open("a") as f:
                    f.write(json.dumps(error) + "\n")
                return
            rows.append(row)
            with (out_dir / "results.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    await asyncio.gather(*(one(case) for case in cases))
    return rows, errors
