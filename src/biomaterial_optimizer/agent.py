"""A PydanticAI agent that answers questions about seaweed films using this project's
data and models, and proposes recipes to try next.

The LLM (Claude) understands the question and decides which tools to call; the numbers
always come from the tools: the database, the model table and the GP-based proposals.

Tools (typed, so Claude can only call them with valid arguments):
  query_database    read-only SQL over recipes / ingredients / properties
  data_coverage     how many recipes and papers back a target (is the data thin?)
  propose_recipes   recipe changes predicted to raise water resistance within limits
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

from .data.dataset import TARGETS, change_vs_control
from .models.propose import propose

MODEL = "anthropic:claude-opus-5-5"
MAX_ROWS = 50

Target = Literal[
    "water_contact_angle",
    "water_solubility",
    "uv_transmittance_280nm",
    "antioxidant_dpph",
    "tensile_strength",
    "elongation_at_break",
]
Base = Literal["alginate", "agar", "carrageenan", "seaweed"]


@dataclass
class Deps:
    db_path: Path = Path("data/biomaterials.duckdb")
    table_path: Path = Path("data/processed/seaweed_biomaterials.parquet")


INSTRUCTIONS = """\
You help materials scientists design seaweed-based films (alginate, agar, carrageenan,
whole seaweed) that resist degradation (water, UV, oxidation, microbes), and explain
what changes cost in strength and flexibility.

Data (DuckDB, read with query_database):
- recipes(recipe_id, name, source)  source is a DOI or materiom:group:<id>
- ingredients(recipe_id, ingredient, amount, unit)  treatments and processing steps
  are ingredients too, e.g. 'calcium chloride bath' (% w/v), 'beeswax coating' (coats),
  'drying temperature' (°C), 'drying time' (h), 'crosslinking time' (min)
- properties(recipe_id, property, value, sd, unit, reported)  value is empty for limits
  and ranges; time points are in the name, e.g. 'soil weight loss day 28'
Names are lowercase, e.g. 'alginate', 'glycerol', 'water contact angle',
'tensile strength', 'elongation at break', 'water solubility'.

Rules:
- Take every number from a tool. Never state a value you did not get from a tool.
- Labs measure differently: values from different papers are not directly comparable.
  Prefer changes within one paper, and say when you compare across papers.
- Check data_coverage before drawing conclusions about a target; say plainly when the
  data is thin (fewer than 3 papers). Microbial resistance and lifespan have almost no
  data.
- For "what should I try" questions, call propose_recipes. Its predictions are GP
  guesses with ranges; always pass on the "untested" notes (changes no paper measured
  for that property) and say proposals must be tested in the lab.
- Answer in plain language for a scientist, briefly. End with a short "Caveats" line.
"""

agent = Agent(
    MODEL,
    deps_type=Deps,
    instructions=INSTRUCTIONS,
    model_settings=AnthropicModelSettings(
        anthropic_effort="high",
        # On a refusal, the API re-runs the request on a fallback model it picks.
        anthropic_betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
    ),
    defer_model_check=True,  # lets tests and imports run without an API key
)


def run_query(db_path: Path, sql: str) -> str:
    """Run one read-only query; no file, network or extension access."""
    con = duckdb.connect(
        str(db_path), read_only=True, config={"enable_external_access": False}
    )
    try:
        df = con.sql(sql).df()
    except duckdb.Error as e:
        return f"Query failed: {e}"
    finally:
        con.close()
    note = f"\n({len(df)} rows, showing {MAX_ROWS})" if len(df) > MAX_ROWS else ""
    return df.head(MAX_ROWS).to_string(index=False) + note


def coverage(table_path: Path, target: str) -> dict:
    absolute = pd.read_parquet(table_path)
    changes = change_vs_control(absolute)
    out = {}
    for name, df in {"absolute": absolute, "change_vs_control": changes}.items():
        rows = df.loc[df[target].notna()]
        out[name] = {"recipes": len(rows), "papers": int(rows["source"].nunique())}
    return out


@agent.tool
def query_database(ctx: RunContext[Deps], sql: str) -> str:
    """Run a read-only SQL query (DuckDB) over the recipes, ingredients and properties
    tables. Returns at most 50 rows as text, or the error message to fix the query."""
    return run_query(ctx.deps.db_path, sql)


@agent.tool
def data_coverage(ctx: RunContext[Deps], target: Target) -> dict:
    """How many seaweed recipes and papers measured `target`, both as absolute values
    and as changes against each paper's own plain control film."""
    return coverage(ctx.deps.table_path, target)


@agent.tool
def propose_recipes(
    ctx: RunContext[Deps],
    base: Base = "alginate",
    max_tensile_loss: float = Field(0.10, ge=0, le=0.9),
    max_elongation_loss: float = Field(0.25, ge=0, le=0.9),
    n: int = Field(5, ge=1, le=10),
) -> list[dict]:
    """Propose changes to a plain `base` film predicted to raise water contact angle
    while tensile strength and elongation fall by at most the given fractions
    (0.10 = 10 %). Each proposal has predicted % changes with a ±1 sd range and an
    'untested' note listing changes no recipe ever tried for that property."""
    proposals = propose(
        change_vs_control(pd.read_parquet(ctx.deps.table_path)),
        base=base,
        max_tensile_loss=max_tensile_loss,
        max_elongation_loss=max_elongation_loss,
        n_proposals=n,
    )
    return proposals.drop(columns="score").to_dict("records")


assert set(Target.__args__) == set(TARGETS)  # keep the tool's choices in sync
