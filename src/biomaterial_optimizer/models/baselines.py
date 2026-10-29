"""Simple baseline models per target, scored with leave-one-source-out cross-validation.

Leave-one-source-out (LOSO): hold out every recipe from one paper, train on all the
others, predict the held-out paper; repeat for each paper. The score answers "how well
does this do on a lab it has never seen?", which matters because values differ more
between labs than within them. The Gaussian process has to beat these numbers.

For comparison, `cv="random"` uses ordinary 5-fold CV, where recipes from the same
paper end up in both training and test data. The gap between the two shows how much
a model is learning "which paper is this" rather than chemistry.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.dataset import ADDITIVE_ROLES, BASE_TYPES, PROCESSING

NUMERIC = [
    *ADDITIVE_ROLES,
    "crosslink_bath",
    "surface_treatment",
    *(column for column, _ in PROCESSING.values()),
]
FEATURES = [*NUMERIC, *(f"base_{b}" for b in BASE_TYPES)]

MODELS = {
    # Always predicts the training mean: the floor every real model must beat.
    "mean": lambda: DummyRegressor(),
    # A straight line per feature; scaled so the penalty treats features equally.
    # Missing drying info is filled with the training median (inside each CV fold,
    # so the held-out recipes never influence it).
    "ridge": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
    ),
    # Many decision trees averaged; can learn curves and interactions.
    "random_forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
    ),
}


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: the numeric amounts plus one 0/1 column per base type."""
    X = df[NUMERIC].astype(float)
    for b in BASE_TYPES:
        X[f"base_{b}"] = (df["base"] == b).astype(float)
    return X.loc[:, FEATURES]


CV = {
    "loso": LeaveOneGroupOut(),
    "random": KFold(5, shuffle=True, random_state=0),
}


def cv_predictions(
    df: pd.DataFrame, target: str, model: str, cv: str = "loso"
) -> np.ndarray:
    """Held-out prediction for every recipe that measured `target`."""
    rows = df.loc[df[target].notna()]
    return np.asarray(
        cross_val_predict(
            MODELS[model](),
            features(rows),
            rows[target],
            groups=rows["source"] if cv == "loso" else None,
            cv=CV[cv],
        )
    )


def evaluate(df: pd.DataFrame, target: str, model: str, cv: str = "loso") -> dict:
    """CV scores for one target and model.

    rmse/mae are in the target's own unit. r2 compares against always guessing the
    overall mean: 1 is perfect, 0 is no better, below 0 is worse.
    """
    rows = df.loc[df[target].notna()]
    y, pred = rows[target], cv_predictions(df, target, model, cv)
    return {
        "n_recipes": len(rows),
        "n_sources": rows["source"].nunique(),
        "rmse": root_mean_squared_error(y, pred),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred),
    }
