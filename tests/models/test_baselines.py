import numpy as np
import pandas as pd
import pytest

from biomaterial_optimizer.models.baselines import (
    FEATURES,
    MODELS,
    cv_predictions,
    evaluate,
    features,
)


def table(n_sources=4, per_source=6, lab_offset=0.0, seed=0):
    """A fake model table where tensile strength = 10 * plasticiser (+ a lab offset)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sources):
        for _ in range(per_source):
            plasticiser = rng.uniform(0, 1)
            rows.append(
                {
                    "source": f"paper{s}",
                    "base": "alginate",
                    "plasticiser": plasticiser,
                    "tensile_strength": 10 * plasticiser + lab_offset * s,
                }
            )
    df = pd.DataFrame(rows)
    for column in FEATURES:
        if column not in df and not column.startswith("base_"):
            df[column] = 0.0
    return df


def test_features_one_hot_base():
    df = table().assign(base=["agar"] + ["alginate"] * 23)
    X = features(df)
    assert list(X.columns) == FEATURES
    assert X["base_agar"].tolist()[:2] == [1.0, 0.0]


def test_ridge_learns_a_clean_signal():
    scores = evaluate(table(), "tensile_strength", "ridge")
    assert scores["r2"] > 0.95
    assert (scores["n_recipes"], scores["n_sources"]) == (24, 4)


def test_loso_never_sees_the_held_out_paper():
    # Each paper's values are shifted by a lab offset the inputs can't explain:
    # random CV can partly memorise it, leave-one-source-out cannot.
    df = table(lab_offset=20.0)
    loso = evaluate(df, "tensile_strength", "random_forest", cv="loso")["r2"]
    random = evaluate(df, "tensile_strength", "random_forest", cv="random")["r2"]
    assert loso < random


def test_rows_without_the_target_are_ignored():
    df = table()
    df.loc[:4, "tensile_strength"] = None
    assert len(cv_predictions(df, "tensile_strength", "mean")) == 19


@pytest.mark.parametrize("model", MODELS)
def test_every_model_runs(model):
    evaluate(table(), "tensile_strength", model)

