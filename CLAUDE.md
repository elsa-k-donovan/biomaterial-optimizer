# biomaterial-optimizer

Portfolio project for an ML role at Materiom. Question it answers:

> Which additives and processing choices make seaweed-based films more resistant
> to the main causes of degradation (water, UV, oxidation, microbes), and what do
> they cost in strength and flexibility?

Pipeline: Materiom recipes + hand-extracted literature → DuckDB → model table →
Gaussian process (written from scratch in PyTorch) → Bayesian optimisation
(BoTorch) → PydanticAI agent. Tracking with MLflow.

The owner has a software/data engineering background and is new to ML/MLOps:
explain ML terms in plain language when they come up.

## Commands

- Install: `uv sync` (add packages with `uv add`, never pip)
- Download Materiom recipes: `uv run --env-file .env scripts/fetch_materiom.py` (`--limit N` for a trial)
- Build the database: `uv run scripts/build_db.py` (always rebuilds from scratch)
- Build the model table: `uv run scripts/build_dataset.py` (run after `build_db.py`)
- Query: `uv run scripts/query.py "SELECT ..."` or `duckdb data/biomaterials.duckdb`
- Tests: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .`

## Layout

- `scripts/` thin entry points; logic lives in `src/biomaterial_optimizer/`
- `src/biomaterial_optimizer/data/schema.py` the three data models + DuckDB table definitions
- `src/biomaterial_optimizer/data/materiom.py` one Materiom JSON → rows
- `src/biomaterial_optimizer/data/build.py` read → clean → check → write
- `src/biomaterial_optimizer/data/dataset.py` database → model table (unit conversion, roles, targets)
- `data/aliases.yaml` all name/unit fixes (lowercase keys)
- `data/ingredient_roles.csv` ingredient → role (+ density for liquids); unlisted → `other`
- `data/processed/` model table + its `skipped.csv` (gitignored, rebuilt by the script)
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

## Model table (`data/processed/seaweed_biomaterials.parquet`)

One row per seaweed recipe (a recipe with a `base` ingredient).

- Inputs: `base` (alginate/agar/carrageenan/seaweed), additive amounts in g per g of
  base polymer summed per role (plasticiser, copolymer, filler, crosslinker,
  water_repellent, active, other), `crosslink_bath` (% w/v), `surface_treatment` (count).
- Targets, each with `_sd`: water contact angle, water solubility, UV transmittance 280 nm,
  DPPH antioxidant activity, tensile strength, elongation at break.
- Also `recipe_id`, `source` (for leave-one-source-out CV) and the `ingredients` list.
- Assumptions (commented in `dataset.py`): a bare `%` on an additive means % of base
  weight; a bare `%` on a bath means % w/v; calcium chloride is a bath.
- Recipes that can't be converted, or measured no target, go to `data/processed/skipped.csv`.
- Targets are picked by data coverage, not by what equipment the owner has.

## Literature (4 papers, notes in data/literature/sources.yaml)

- surya2022: seaweed + kenaf cellulose fibre + silane; the only numeric soil-burial data
- garcia: alginate + recycled PVA (synthetic) + beeswax; outdoor durability is qualitative only
- abdollahi2013: alginate + cellulose nanocrystals; water resistance, no lifespan
- liling2016: alginate crosslinked with Ca/Zn/Mn/Al; crosslinking time is only in the recipe name

Do not commit PDFs (two are not open access), `.env`, `data/raw/` or `*.duckdb`.

## Known data issues

- Water vapour permeability appears in two units (`g mm/m2 d kPa`, `g/(m s Pa)`); convert in the modeling step.
- Values differ more between batches/labs than within them: validate leave-one-source-out.
- Microbes/biodegradation has too little data (16 seaweed recipes, 2 papers) to model;
  treat it as a documented data gap unless more papers are added.
- One Materiom paper (`10.15376/biores.14.2.3389-3410`) has wrongly entered cellulose
  amounts and is excluded in `dataset.py`.

## Plan

1. ~~Model table~~ (done: `scripts/build_dataset.py`).
2. Baselines (ridge, random forest) per target with leave-one-source-out CV, logged to MLflow.
3. Multi-output Gaussian process from scratch in PyTorch (all targets together, so
   well-measured targets help thin ones), using each measurement's sd as its noise.
   **The owner writes this part herself; review and explain, don't write it.**
4. Chemical language model embeddings (e.g. ChemBERTa) of additives as GP features,
   so the model can reason about unseen additives.
5. Bayesian optimisation (BoTorch): pool-based benchmark vs random, then constrained
   recipe proposals (amounts within seen ranges, strength above a minimum).
6. MLOps: GitHub Actions running tests, MLflow model registry, data checks,
   PydanticAI agent with typed tools and evals.

Closed loop (later, once the owner has lab access): her own results go in `data/lab/`
as another source (`lab:<name>`, ids `lab-001`) in the same three tables; retrain and
let BO propose the next recipe. First repeat 1–2 paper recipes so the model can learn
the lab's offset; make each recipe 3 times (mean → value, spread → sd).

## Working style

- Keep everything as simple as possible; prefer fewer files and fewer columns.
- Plain-language explanations; short answers.
- Small steps, tests alongside code, run `uv run pytest` before finishing.
- Ask before redesigns or anything that changes the data model.
