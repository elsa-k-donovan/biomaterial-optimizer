"""Propose recipe changes that should raise water resistance (contact angle) while
keeping the loss in strength and flexibility within limits; logs to MLflow.

Run build_dataset.py first.

Usage:
    uv run scripts/propose_recipes.py
    uv run scripts/propose_recipes.py --base agar --max-tensile-loss 0.05
"""

import argparse
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from biomaterial_optimizer.data.dataset import BASE_TYPES, change_vs_control
from biomaterial_optimizer.models.propose import propose

TABLE = Path("data/processed/seaweed_biomaterials.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else ""
    )
    parser.add_argument("--base", choices=BASE_TYPES, default="alginate")
    parser.add_argument("--max-tensile-loss", type=float, default=0.10)
    parser.add_argument("--max-elongation-loss", type=float, default=0.25)
    parser.add_argument("-n", type=int, default=5, help="number of proposals")
    args = parser.parse_args()

    proposals = propose(
        change_vs_control(pd.read_parquet(TABLE)),
        base=args.base,
        max_tensile_loss=args.max_tensile_loss,
        max_elongation_loss=args.max_elongation_loss,
        n_proposals=args.n,
    )
    print(f"Changes to a plain {args.base} film (predicted %, ±1 sd range):\n")
    for i, p in enumerate(proposals.to_dict("records"), 1):
        print(f"{i}. {p['changes']}")
        for column in proposals.columns:
            if column.endswith(" %"):
                print(f"     {column[:-2]:<22} {p[column]}")
        if p["untested"]:
            print(f"     never tried: {p['untested']}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("proposals")
    with mlflow.start_run(run_name=args.base):
        mlflow.log_params(vars(args) | {"surrogate": "botorch_gp"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposals.csv"
            proposals.to_csv(path, index=False)
            mlflow.log_artifact(str(path))


if __name__ == "__main__":
    main()
