"""Score baseline models on every target with cross-validation; log to MLflow.

Each model is scored on two versions of the model table:
  absolute  the measured values (e.g. 22 MPa)
  change    change vs each paper's own control recipe (e.g. +24 %), see
            change_vs_control(); it cancels out how each lab measures
and with two kinds of CV: leave-one-source-out (the honest score) and random
5-fold (for comparison; it leaks lab effects between training and test data).
Targets measured in fewer than MIN_SOURCES papers are skipped.

Run build_dataset.py first.

Usage:
    uv run scripts/train_baselines.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
"""

import hashlib
from pathlib import Path

import mlflow
import pandas as pd

from biomaterial_optimizer.data.dataset import TARGETS, change_vs_control
from biomaterial_optimizer.models.baselines import CV, MIN_SOURCES, MODELS, evaluate

TABLE = Path("data/processed/seaweed_biomaterials.parquet")


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baselines")
    absolute = pd.read_parquet(TABLE)
    tables = {"absolute": absolute, "change": change_vs_control(absolute)}
    # Which exact table the runs used, so results can be traced back to the data.
    table_md5 = hashlib.md5(TABLE.read_bytes()).hexdigest()

    print(f"{'table':<9} {'target':<24} {'model':<14} {'n':>4} {'src':>4}", end="")
    print("".join(f" {cv + ' r2':>10}" for cv in CV))
    for name, df in tables.items():
        for target in TARGETS:
            n_sources = df.loc[df[target].notna(), "source"].nunique()
            if n_sources < MIN_SOURCES:
                print(f"{name:<9} {target:<24} skipped: only {n_sources} papers")
                continue
            for model in MODELS:
                print(f"{name:<9} {target:<24} {model:<14}", end="")
                for cv in CV:
                    scores = evaluate(df, target, model, cv)
                    params = {"table": name, "target": target, "model": model, "cv": cv}
                    log_run(params, scores, table_md5)
                    if cv == "loso":
                        print(
                            f" {scores['n_recipes']:>4} {scores['n_sources']:>4}",
                            end="",
                        )
                    print(f" {scores['r2']:>10.2f}", end="")
                print()


def log_run(params: dict, scores: dict, table_md5: str) -> None:
    with mlflow.start_run(run_name=" / ".join(params.values())):
        mlflow.log_params(
            params
            | {
                "n_recipes": scores["n_recipes"],
                "n_sources": scores["n_sources"],
                "table_md5": table_md5,
            }
        )
        mlflow.log_metrics({k: scores[k] for k in ("rmse", "mae", "r2")})


if __name__ == "__main__":
    main()
