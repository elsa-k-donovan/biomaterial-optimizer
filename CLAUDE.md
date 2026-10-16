# biomaterial-optimizer

Portfolio project for an ML role at Materiom. Question it answers:

> Which ingredients and processing choices make seaweed-based biomaterials more
> water-resistant (a stand-in for longevity), and what do they cost in strength
> and flexibility?

Pipeline: Materiom recipes + hand-extracted literature → DuckDB → model table →
Gaussian process (written from scratch in PyTorch) → Bayesian optimisation
(BoTorch) → PydanticAI agent. Tracking with MLflow.

## Commands

- Install: `uv sync` (add packages with `uv add`, never pip)
- Download Materiom recipes: `uv run --env-file .env scripts/fetch_materiom.py` (`--limit N` for a trial)
- Build the database: `uv run scripts/build_db.py` (always rebuilds from scratch)
- Query: `uv run scripts/query.py "SELECT ..."` or `duckdb data/biomaterials.duckdb`
- Tests: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .`

## Layout

- `scripts/` thin entry points; logic lives in `src/biomaterial_optimizer/`
- `src/biomaterial_optimizer/data/schema.py` the three data models + DuckDB table definitions
- `src/biomaterial_optimizer/data/materiom.py` one Materiom JSON → rows
- `src/biomaterial_optimizer/data/build.py` read → clean → check → write
- `data/aliases.yaml` all name/unit fixes (lowercase keys)
- `data/literature/` `recipes.csv`, `ingredients.csv`, `properties.csv`, `sources.yaml`
- `data/raw/materiom/recipes/` downloaded JSON cache (gitignored)
- `tests/` pytest; shared fixtures in `tests/conftest.py`

## Data model (keep it this simple)

Three tables, same shape for Materiom and literature:

- `recipes`: recipe_id, name, source (DOI, or `materiom:group:<id>`)
- `ingredients`: recipe_id, ingredient, amount, unit
- `properties`: recipe_id, property, value, sd, unit, reported

Rules:
- `reported` always holds exactly what the source said.
- `value`/`sd` are filled only for plain numbers or "x ± y". Limits (`<0.1`) and
  ranges (`20–30`) are never converted to a number; `value` stays empty.
- Time points go in the property name, e.g. `soil weight loss day 28`.
- Treatments are ingredients, e.g. `silane surface treatment`, `beeswax coating` (unit: coats),
  `calcium chloride bath` (unit: % w/v).
- Invalid rows go to `data/skipped.csv` with a reason, never dropped silently.
- Literature recipe ids are prefixed with the paper: `surya2022-SF4-S`.
- A paper already in Materiom is not loaded again from the literature (no double counting).
- Seaweed-only filtering happens in the modeling step, not in the database.

## Literature (4 papers, notes in data/literature/sources.yaml)

- surya2022: seaweed + kenaf cellulose fibre + silane; the only numeric soil-burial data
- garcia: alginate + recycled PVA (synthetic) + beeswax; outdoor durability is qualitative only
- abdollahi2013: alginate + cellulose nanocrystals; water resistance, no lifespan
- liling2016: alginate crosslinked with Ca/Zn/Mn/Al; crosslinking time is only in the recipe name

Do not commit PDFs (two are not open access), `.env`, `data/raw/` or `*.duckdb`.

## Known data issues

- Water vapour permeability appears in two units (`g mm/m2 d kPa`, `g/(m s Pa)`); convert in the modeling step.
- Values differ more between batches/labs than within them: validate leave-one-source-out.

## Next steps

1. `scripts/build_dataset.py`: map ingredients to roles (base polymer, plasticiser, filler,
   water repellent, crosslinker, surface treatment) via a list in `data/`, express amounts
   per gram of base polymer, filter to seaweed bases, write one row per recipe to
   `data/processed/seaweed_biomaterials.parquet`.
2. Baselines with leave-one-source-out CV, logged to MLflow.
3. Gaussian process from scratch in PyTorch, using each measurement's sd as its noise.
   **The owner writes this part herself; review and explain, don't write it.**
4. Pool-based BO benchmark vs random, then constrained recipe proposals.
5. PydanticAI agent with typed tools and evals.

## Working style

- Keep everything as simple as possible; prefer fewer files and fewer columns.
- Plain-language explanations; short answers.
- Small steps, tests alongside code, run `uv run pytest` before finishing.
- Ask before redesigns or anything that changes the data model.
