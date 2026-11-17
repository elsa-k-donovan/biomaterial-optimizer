"""Pool-based benchmark: Bayesian optimisation vs random picking; logs to MLflow.

Uses the change-vs-control table, so "best" means the biggest improvement over a
paper's own plain film. Each run starts from N_INIT random recipes, then picks BUDGET
more; repeated with REPEATS different starting sets (the same ones for both strategies).

Run build_dataset.py first.

Usage:
    uv run scripts/bo_benchmark.py
"""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from biomaterial_optimizer.data.dataset import change_vs_control
from biomaterial_optimizer.models.baselines import features
from biomaterial_optimizer.models.bo import run, scale

TABLE = Path("data/processed/seaweed_biomaterials.parquet")
TARGETS = ["water_contact_angle", "tensile_strength", "elongation_at_break"]
N_INIT, BUDGET, REPEATS = 5, 25, 20
STRATEGIES = ["bo", "random"]


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bo_benchmark")
    df = change_vs_control(pd.read_parquet(TABLE))

    print(f"{'target':<22} {'pool':>4}  rounds to top 5 % (mean, BO vs random)")
    for target in TARGETS:
        rows = df.loc[df[target].notna()]
        X, y = scale(features(rows).to_numpy(float)), rows[target].to_numpy()
        top5 = np.quantile(y, 0.95)
        curves = {
            s: np.array(
                [
                    run(X, y, s, N_INIT, BUDGET, np.random.default_rng(seed))
                    for seed in range(REPEATS)
                ]
            )
            for s in STRATEGIES
        }
        # First pick (after the starting set) whose best-so-far is in the top 5 %;
        # BUDGET + 1 when it never gets there.
        rounds = {
            s: np.where(
                (c[:, N_INIT - 1 :] >= top5).any(axis=1),
                (c[:, N_INIT - 1 :] >= top5).argmax(axis=1),
                BUDGET + 1,
            )
            for s, c in curves.items()
        }
        log(target, len(y), curves, rounds)
        print(
            f"{target:<22} {len(y):>4}  "
            f"{rounds['bo'].mean():5.1f} vs {rounds['random'].mean():5.1f}"
        )


def log(target: str, n_pool: int, curves: dict, rounds: dict) -> None:
    with mlflow.start_run(run_name=target):
        mlflow.log_params(
            {
                "target": target,
                "table": "change",
                "n_pool": n_pool,
                "n_init": N_INIT,
                "budget": BUDGET,
                "repeats": REPEATS,
                "surrogate": "botorch_gp",
                "acquisition": "expected_improvement",
            }
        )
        for s in STRATEGIES:
            mlflow.log_metric(f"{s}_rounds_to_top5", float(rounds[s].mean()))
            mlflow.log_metric(
                f"{s}_best_after_10", float(curves[s][:, N_INIT + 9].mean())
            )
        # Mean best-so-far after each pick, per strategy: the learning curve.
        table = pd.DataFrame({s: c.mean(axis=0) for s, c in curves.items()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"curves_{target}.csv"
            table.rename_axis("recipes_made").to_csv(path)
            mlflow.log_artifact(str(path))


if __name__ == "__main__":
    main()
