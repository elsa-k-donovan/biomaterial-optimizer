"""Build data/biomaterials.duckdb from downloaded Materiom recipes and the literature CSVs.

Always rebuilds from scratch, so rerun it whenever the data or aliases change.

Usage:
    uv run scripts/build_db.py
"""

from collections import Counter
from pathlib import Path

from biomaterial_optimizer.data.build import build_database

DB = Path("data/biomaterials.duckdb")


def main() -> None:
    counts, skipped = build_database(
        db_path=DB,
        materiom_dir=Path("data/raw/materiom/recipes"),
        literature_dir=Path("data/literature"),
        aliases_path=Path("data/aliases.yaml"),
    )
    print(f"Built {DB}:")
    for table, n in counts.items():
        print(f"  {table:<12} {n:>6} rows")
    if skipped:
        print(f"Skipped {len(skipped)} rows (details in {DB.parent / 'skipped.csv'}):")
        for (table, reason), n in Counter(
            (s["table"], s["reason"]) for s in skipped
        ).most_common():
            print(f"  {n:>5}  {table}: {reason}")


if __name__ == "__main__":
    main()
