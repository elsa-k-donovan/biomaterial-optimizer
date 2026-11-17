"""Propose new recipes: changes to a plain film that should make it more water-resistant
without costing too much strength or flexibility.

Works on the change-vs-control table, so a proposal reads like a lab instruction:
"starting from your plain alginate film, add 0.3 g filler per g alginate and dip it in
a 2 % calcium bath", with predicted % changes in water contact angle, tensile strength
and elongation.

How:
1. Fit one GP per target on the recipes that measured it.
2. Generate many candidate recipes: 1-3 changes each, every amount within the range
   seen in the data (the GP can't be trusted far outside it).
3. Score each candidate with constrained Expected Improvement:
   (expected gain in contact angle over the best seen) x
   (chance tensile strength drops by no more than the limit) x
   (chance elongation drops by no more than the limit).
4. Return the top proposals, each changing a different set of things.

These are hypotheses to test, not guarantees: leave-one-source-out showed the GP can't
predict levers that only one paper has tried.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..data.dataset import INPUTS
from .baselines import features
from .bo import Surrogate, botorch_gp, expected_improvement, scale

OBJECTIVE = "water_contact_angle"
CONSTRAINTS = ["tensile_strength", "elongation_at_break"]

# How each input change is described in a proposal.
LABELS = {
    "crosslink_bath": "% w/v crosslinking bath",
    "surface_treatment": "coatings / surface treatments",
    "drying_temperature": "°C drying temperature",
    "drying_time": "h drying time",
    "crosslinking_time": "min in the bath",
}


def candidates(
    changes: pd.DataFrame, base: str, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """`n` random candidate recipes, each changing 1-3 inputs within their seen range.

    Coatings come in whole numbers, and bath time is only set together with a bath.
    """
    seen = changes[INPUTS]
    low, high = seen.min(), seen.max()
    levers = [c for c in INPUTS if high[c] > low[c] and c != "crosslinking_time"]
    rows = []
    for _ in range(n):
        row = dict.fromkeys(INPUTS, 0.0)
        for c in rng.choice(levers, rng.integers(1, 4), replace=False):
            if c == "surface_treatment":
                row[c] = float(rng.integers(max(1, low[c]), high[c] + 1))
            else:
                row[c] = float(rng.uniform(low[c], high[c]))
            if c == "crosslink_bath":
                t = "crosslinking_time"
                row[t] = float(rng.uniform(max(0.0, low[t]), high[t]))
        rows.append(row)
    return pd.DataFrame(rows).assign(base=base)


def describe(row: dict) -> str:
    parts = []
    for c in INPUTS:
        if row[c] != 0:
            unit = LABELS.get(c, f"g {c.replace('_', ' ')} per g base")
            parts.append(f"{row[c]:+.2f} {unit}")
    return "; ".join(parts)


def propose(
    changes: pd.DataFrame,
    base: str = "alginate",
    max_tensile_loss: float = 0.10,
    max_elongation_loss: float = 0.25,
    n_candidates: int = 5000,
    n_proposals: int = 5,
    rng: np.random.Generator | None = None,
    surrogate: Surrogate = botorch_gp,
) -> pd.DataFrame:
    """Top proposals with predicted % changes (and a ±1 sd range) for each target.

    `changes` is the change-vs-control table. Losses are fractions: 0.10 allows
    tensile strength to fall by at most 10 %.
    """
    rng = rng or np.random.default_rng(0)
    cands = candidates(changes, base, n_candidates, rng)
    X_all = features(changes).to_numpy(float)
    X_cand = scale(features(cands).to_numpy(float), ref=X_all)

    predicted, tried = {}, {}
    for target in [OBJECTIVE, *CONSTRAINTS]:
        rows = changes.loc[changes[target].notna()]
        tried[target] = {c for c in INPUTS if rows[c].fillna(0).ne(0).any()}
        X = scale(features(rows).to_numpy(float), ref=X_all)
        y = rows[target].to_numpy(float)
        predicted[target] = (*surrogate(X, y, X_cand), y.max())

    mean, std, best = predicted[OBJECTIVE]
    score = expected_improvement(mean, std, best)
    limits = dict(zip(CONSTRAINTS, (max_tensile_loss, max_elongation_loss)))
    for target, loss in limits.items():
        m, s, _ = predicted[target]
        score = score * norm.cdf((m - np.log(1 - loss)) / s)  # P(drop <= loss)

    out = cands.assign(score=score).sort_values("score", ascending=False)
    out["changes"] = [describe(r) for r in out.to_dict("records")]
    out["levers"] = out[INPUTS].ne(0).apply(lambda r: tuple(r[r].index), axis=1)
    out = out.drop_duplicates("levers").head(n_proposals)

    # log-ratio -> % change, with a ±1 sd range.
    for target, (m, s, _) in predicted.items():
        m, s = (
            pd.Series(m, cands.index)[out.index],
            pd.Series(s, cands.index)[out.index],
        )
        out[f"{target} %"] = [
            f"{np.expm1(a) * 100:+.0f} ({np.expm1(a - b) * 100:+.0f} to {np.expm1(a + b) * 100:+.0f})"
            for a, b in zip(m, s)
        ]
    # A change no recipe ever tried for a target means that prediction is a guess.
    out["untested"] = [
        "; ".join(
            f"{c} for {t}" for t in predicted for c in levers if c not in tried[t]
        )
        for levers in out["levers"]
    ]
    columns = ["changes", *(f"{t} %" for t in predicted), "untested", "score"]
    return out.loc[:, columns].reset_index(drop=True)
