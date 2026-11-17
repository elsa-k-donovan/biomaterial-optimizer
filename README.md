# biomaterial-optimizer 🧫

![CI](https://github.com/elsa-k-donovan/biomaterial-optimizer/actions/workflows/ci.yml/badge.svg)

Machine learning for seaweed-based films: **which ingredients make them last longer, what
that costs in strength and flexibility, and which recipe to test next.**

## How it works

1. **Data:** 700 recipes from [Materiom](https://materiom.org) and 4 papers, cleaned into
   a DuckDB database.
2. **Model table:** 158 seaweed recipes, with ingredient amounts per gram of seaweed and
   6 measured properties (e.g. water resistance, strength).
3. **Baselines:** simple models scored per lab, logged to MLflow.
4. **Bayesian optimisation:** a Gaussian process (BoTorch) picks the next recipe to test.
5. **Recipe proposals:** suggested changes to a plain film, with predicted effects.
6. **Agent:** ask questions in plain English; Claude answers using the data and models.

## Findings

- **Labs differ more than recipes.** The same plain film measures 18 MPa in one lab and
  40 MPa in another. Models can't yet predict a lab they haven't seen.
- **Bayesian optimisation finds good recipes 2–3× faster than random picking:**

  | Target | Random | Bayesian optimisation |
  |---|---|---|
  | Water resistance | 7.5 picks | **4.1** |
  | Strength | 11.3 | **3.6** |
  | Flexibility | 6.2 | **3.5** |

  *Picks needed to find a top-5 % recipe, average of 20 runs.*

- **Lifespan is barely measured.** Only one paper has soil-burial data, so water
  resistance stands in for longevity.

## Run it

Needs [uv](https://docs.astral.sh/uv/). Materiom and Anthropic API keys go in `.env`.

```bash
uv sync
uv run pytest                            # tests
uv run scripts/build_db.py               # build the database
uv run scripts/build_dataset.py          # build the model table
uv run scripts/bo_benchmark.py           # BO vs random
uv run scripts/propose_recipes.py        # recipe proposals
uv run --env-file .env scripts/agent.py "How can I make alginate film more water-resistant?"
```

Downloading Materiom data: `uv run --env-file .env scripts/fetch_materiom.py`.

## Next steps

- More papers that test the same ingredients, to learn across labs.
- Chemical embeddings, so the model knows *which* additive, not just its role.
- A closed loop: add new lab results, retrain, propose again.

## Credits

Recipe data from [Materiom Commons](https://materiom.org), used non-commercially.
Literature sources in `data/literature/sources.yaml`.
