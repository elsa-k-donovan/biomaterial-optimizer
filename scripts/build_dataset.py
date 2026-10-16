"""Build the model table data/processed/seaweed_biomaterials.parquet from the database.

Run build_db.py first. Rerun whenever the database or data/ingredient_roles.csv changes.

Usage:
    uv run scripts/build_dataset.py
"""

from collections import Counter
from pathlib import Path

import duckdb

from biomaterial_optimizer.data.dataset import TARGETS, build_dataset

OUT = Path("data/processed/seaweed_biomaterials.parquet")


def main() -> None:
    n, skipped, unlisted = build_dataset(
        db_path=Path("data/biomaterials.duckdb"),
        roles_path=Path("data/ingredient_roles.csv"),
        out_path=OUT,
    )
    print(f"Built {OUT}: {n} seaweed recipes")
    counts = (
        duckdb.sql(
            f"SELECT {', '.join(f'count({t})' for t in TARGETS)} FROM '{OUT}'"
        ).fetchone()
        or ()
    )
    for target, count in zip(TARGETS, counts):
        print(f"  {target:<24} {count:>4} measured")
    if skipped:
        print(
            f"Skipped {len(skipped)} recipes (details in {OUT.parent / 'skipped.csv'}):"
        )
        for reason, count in Counter(s["reason"] for s in skipped).most_common():
            print(f"  {count:>5}  {reason}")
    if unlisted:
        print(f"{len(unlisted)} ingredients have no role, counted as 'other':")
        print("  " + ", ".join(sorted(unlisted)))


if __name__ == "__main__":
    main()
