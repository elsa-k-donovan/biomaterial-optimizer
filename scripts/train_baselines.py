"""Score baseline models on every target with cross-validation; log to MLflow.

Each model is scored twice: leave-one-source-out (the honest score) and random
5-fold (for comparison; it leaks lab effects between training and test data).

Run build_dataset.py first.

Usage:
    uv run scripts/train_baselines.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
"""

import hashlib
from pathlib import Path

import mlflow
import pandas as pd

from biomaterial_optimizer.data.dataset import TARGETS
from biomaterial_optimizer.models.baselines import CV, MODELS, evaluate

TABLE = Path("data/processed/seaweed_biomaterials.parquet")


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baselines")
    df = pd.read_parquet(TABLE)
    # Which exact table the runs used, so results can be traced back to the data.
    table_md5 = hashlib.md5(TABLE.read_bytes()).hexdigest()

    print(f"{'target':<24} {'model':<14} {'n':>4} {'src':>4}", end="")
    print("".join(f" {cv + ' r2':>10}" for cv in CV))
    for target in TARGETS:
        for model in MODELS:
            print(f"{target:<24} {model:<14}", end="")
            for cv in CV:
                scores = evaluate(df, target, model, cv)
                log_run(target, model, cv, scores, table_md5)
                if cv == "loso":
                    print(f" {scores['n_recipes']:>4} {scores['n_sources']:>4}", end="")
                print(f" {scores['r2']:>10.2f}", end="")
            print()


def log_run(target: str, model: str, cv: str, scores: dict, table_md5: str) -> None:
    with mlflow.start_run(run_name=f"{target} / {model} / {cv}"):
        mlflow.log_params(
            {
                "target": target,
                "model": model,
                "cv": cv,
                "n_recipes": scores["n_recipes"],
                "n_sources": scores["n_sources"],
                "table_md5": table_md5,
            }
        )
        mlflow.log_metrics({k: scores[k] for k in ("rmse", "mae", "r2")})


if __name__ == "__main__":
    main()
