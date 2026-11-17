import numpy as np
import pandas as pd

from biomaterial_optimizer.data.dataset import INPUTS, TARGETS
from biomaterial_optimizer.models.baselines import FEATURES
from biomaterial_optimizer.models.propose import candidates, propose


def change_table():
    """A tiny change-vs-control table where filler, plasticiser and baths were tried."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        row = dict.fromkeys(INPUTS, 0.0)
        # Like real changes vs a control: each lever is 0 in some recipes.
        on = i % 2 == 0
        row |= {"filler": rng.uniform(0, 0.5) * on, "plasticiser": 0.3 * (i % 3 == 0)}
        bath = rng.uniform(0.5, 3) * (i % 4 == 0)
        row |= {"crosslink_bath": bath, "crosslinking_time": 2.0 * (bath > 0)}
        row |= {"surface_treatment": float(i % 3)}
        for t in TARGETS:
            row[t], row[f"{t}_sd"] = rng.normal(0, 0.1), np.nan
        rows.append(row | {"recipe_id": f"r{i}", "source": f"p{i % 4}"})
    return pd.DataFrame(rows).assign(base="alginate")


def test_candidates_stay_in_range_and_make_sense():
    table = change_table()
    cands = candidates(table, "agar", 300, np.random.default_rng(0))
    assert (cands[INPUTS] >= table[INPUTS].min() - 1e-12).all().all()
    assert (cands[INPUTS] <= table[INPUTS].max() + 1e-12).all().all()
    changed = cands[[c for c in INPUTS if c != "crosslinking_time"]].ne(0).sum(axis=1)
    assert changed.between(1, 3).all()
    coats = cands["surface_treatment"]
    assert (coats == coats.round()).all()  # whole coatings only
    no_bath = cands["crosslink_bath"] == 0
    assert (cands.loc[no_bath, "crosslinking_time"] == 0).all()  # time needs a bath
    assert (cands["base"] == "agar").all()


def test_propose_follows_the_model_and_the_limits():
    filler, plasticiser = FEATURES.index("filler"), FEATURES.index("plasticiser")
    calls = []

    def fake_gp(X_seen, y_seen, X_pool):
        # Called for contact angle, then tensile, then elongation.
        calls.append(len(calls))
        if len(calls) == 1:  # filler and plasticiser both raise contact angle
            mean = X_pool[:, filler] + X_pool[:, plasticiser]
        elif len(calls) == 2:  # plasticiser badly weakens the film
            mean = -2.0 * X_pool[:, plasticiser]
        else:
            mean = np.zeros(len(X_pool))
        return mean, np.full(len(X_pool), 0.05)

    out = propose(change_table(), n_candidates=500, n_proposals=3, surrogate=fake_gp)
    assert len(out) == 3
    assert "filler" in out.loc[0, "changes"]
    assert "plasticiser" not in out.loc[0, "changes"]  # costs too much strength
    assert out["changes"].is_unique


def test_flags_changes_never_tried_for_a_target():
    table = change_table()
    table["water_contact_angle"] = np.where(table["surface_treatment"] > 0, np.nan, 0.1)
    out = propose(table, n_candidates=300, n_proposals=10)
    flagged = out.loc[out["changes"].str.contains("coatings")]
    assert (
        flagged["untested"]
        .str.contains("surface_treatment for water_contact_angle")
        .all()
    )
