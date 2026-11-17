"""Run the agent eval (evals/agent_cases.yaml) and log the score to MLflow.

Calls Claude for real, so it costs money. Needs ANTHROPIC_API_KEY in .env, and
build_db.py + build_dataset.py run first.

Usage:
    uv run --env-file .env scripts/eval_agent.py                      # all cases
    uv run --env-file .env scripts/eval_agent.py --cases one-recipe-value,coverage-microbes
"""

import argparse
import asyncio
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import mlflow

from biomaterial_optimizer.agent import MODEL, Deps
from biomaterial_optimizer.agent_eval import load_cases, run_all

CASES = Path("evals/agent_cases.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    args = parser.parse_args()

    cases = load_cases(CASES)
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases.split(",")]
    out_dir = Path("evals/runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    rows, errors = asyncio.run(run_all(cases, Deps(), out_dir))

    for row in sorted(rows, key=lambda r: r["prompt_id"]):
        failed = [name for name, ok in row["checks"].items() if not ok]
        print(
            f"{'PASS' if row['grade'] else 'FAIL'}  {row['prompt_id']:<24} {', '.join(failed)}"
        )
    for error in errors:
        print(f"ERROR {error['prompt_id']:<24} {error['error']}")

    n = len(rows)
    passed = sum(r["grade"] for r in rows)
    rate = passed / n if n else 0.0
    # With few cases one flaky answer moves the score a lot; this shows how much.
    margin = 1.96 * math.sqrt(rate * (1 - rate) / n) if n else 0.0
    tokens_in = sum(r["usage"]["input_tokens"] for r in rows)
    tokens_out = sum(r["usage"]["output_tokens"] for r in rows)
    print(
        f"\n{passed}/{n} passed ({rate:.0%} ± {margin:.0%}), {len(errors)} errors; "
        f"{tokens_in:,} input + {tokens_out:,} output tokens; results in {out_dir}"
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("agent_eval")
    with mlflow.start_run(run_name=out_dir.name):
        mlflow.log_params(
            {"model": MODEL, "n_cases": len(cases), "cases_file": str(CASES)}
        )
        mlflow.log_metrics(
            {
                "pass_rate": rate,
                "errors": len(errors),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
            }
        )
        mlflow.log_artifacts(str(out_dir))


if __name__ == "__main__":
    main()
