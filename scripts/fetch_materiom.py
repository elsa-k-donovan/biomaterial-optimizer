"""Download Materiom Commons recipes, one raw JSON file per recipe.

This script only downloads. `build_db.py` turns the files into tables.
Recipes already on disk are skipped, so rerunning only fetches what is missing.

Needs a .env file in the repo root with:
    MATERIOM_ALGOLIA_APP_ID, MATERIOM_ALGOLIA_API_KEY, MATERIOM_ALGOLIA_INDEX

Usage:
    uv run --env-file .env scripts/fetch_materiom.py            # everything
    uv run --env-file .env scripts/fetch_materiom.py --limit 10 # trial run

Data: Materiom Commons. Credit the source and keep use non-commercial.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

RAW_DIR = Path("data/raw/materiom/recipes")
RECIPE_URL = "https://commons-backend.materiom.org/api/materials-database/recipes/{}"
PAUSE_S = 1.0  # one request per second, to be polite


def list_recipe_ids(client: httpx.Client) -> list[str]:
    """Page through Materiom's search index to get every recipe id."""
    names = [
        "MATERIOM_ALGOLIA_APP_ID",
        "MATERIOM_ALGOLIA_API_KEY",
        "MATERIOM_ALGOLIA_INDEX",
    ]
    if missing := [n for n in names if not os.environ.get(n)]:
        sys.exit(f"Missing {', '.join(missing)}. Run with: uv run --env-file .env ...")
    app_id, api_key, index = (os.environ[n] for n in names)

    url = f"https://{app_id}-dsn.algolia.net/1/indexes/{index}/query"
    headers = {"X-Algolia-Application-Id": app_id, "X-Algolia-API-Key": api_key}
    ids, page = [], 0
    while True:
        response = client.post(
            url, headers=headers, json={"query": "", "page": page, "hitsPerPage": 100}
        )
        response.raise_for_status()
        data = response.json()
        ids += [hit["objectID"] for hit in data["hits"]]
        page += 1
        if page >= data["nbPages"]:
            return ids
        time.sleep(PAUSE_S)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--limit", type=int, help="download at most N new recipes")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30) as client:
        ids = list_recipe_ids(client)
        missing = [i for i in ids if not (RAW_DIR / f"{i}.json").exists()]
        todo = missing[: args.limit] if args.limit else missing
        print(
            f"{len(ids)} recipes listed, {len(ids) - len(missing)} on disk, fetching {len(todo)}"
        )

        failed = []
        for n, recipe_id in enumerate(todo, 1):
            response = client.get(RECIPE_URL.format(recipe_id))
            if response.status_code == 200:
                (RAW_DIR / f"{recipe_id}.json").write_text(response.text)
            else:
                failed.append(recipe_id)
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")
            time.sleep(PAUSE_S)

    if failed:
        print(f"{len(failed)} failed; rerun to retry them.")


if __name__ == "__main__":
    main()
