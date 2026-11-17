"""Pool-based Bayesian optimisation benchmark: does BO find good recipes faster than random?

"Pool-based" means we pretend the recipes in the data haven't been made yet. Each round,
the strategy picks one recipe from the pool, "makes" it (looks up its measured value)
and learns from the result. Bayesian optimisation picks with a Gaussian process (GP):
the GP predicts every remaining recipe's value *and how unsure it is*, and Expected
Improvement favours recipes that are either predicted to be good or very uncertain.
Random picks blindly. A good strategy reaches the best recipes in fewer rounds.

The GP is passed in as a function, so the owner's own GP can replace BoTorch's later:
    surrogate(X_seen, y_seen, X_pool) -> (mean, std) for every pool recipe
"""

from collections.abc import Callable

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm

Surrogate = Callable[
    [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]


def botorch_gp(
    X_seen: np.ndarray, y_seen: np.ndarray, X_pool: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """BoTorch's standard GP: fit on the recipes seen so far, predict the pool."""
    train_X = torch.tensor(X_seen, dtype=torch.float64)
    train_Y = torch.tensor(y_seen, dtype=torch.float64).unsqueeze(-1)
    model = SingleTaskGP(train_X, train_Y)  # standardises y itself
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    with torch.no_grad():
        posterior = model.posterior(torch.tensor(X_pool, dtype=torch.float64))
    mean = posterior.mean.squeeze(-1).numpy()
    std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
    return mean, std


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    """How much each recipe is expected to beat `best`, counting only the upside.

    A recipe scores high if its prediction is above `best` (exploit) or if the GP is
    unsure about it, so it *might* be far above (explore).
    """
    z = (mean - best) / std
    return std * (z * norm.cdf(z) + norm.pdf(z))


def run(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    n_init: int,
    budget: int,
    rng: np.random.Generator,
    surrogate: Surrogate = botorch_gp,
) -> np.ndarray:
    """Best value found after each pick (starting set included), for one run.

    Starts from `n_init` random recipes, then makes `budget` more picks with
    `strategy` ("bo" or "random"). Higher y is better.
    """
    seen = list(rng.choice(len(y), n_init, replace=False))
    for _ in range(budget):
        pool = np.setdiff1d(np.arange(len(y)), seen)
        if not len(pool):
            break
        if strategy == "random":
            pick = rng.choice(pool)
        else:
            mean, std = surrogate(X[seen], y[seen], X[pool])
            pick = pool[np.argmax(expected_improvement(mean, std, y[seen].max()))]
        seen.append(int(pick))
    return np.maximum.accumulate(y[seen])


def scale(X: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Fill gaps with the column median and squeeze each column into 0..1.

    The GP compares recipes by distance, so every input needs the same range.
    Columns that never vary become all 0. Medians and ranges come from `ref`
    (default: X itself), so new recipes can be put on the data's scale.
    """
    ref = X if ref is None else ref
    known = ~np.isnan(ref)
    median = np.array(
        [
            np.median(ref[known[:, j], j]) if known[:, j].any() else 0.0
            for j in range(ref.shape[1])
        ]
    )
    ref = np.where(np.isnan(ref), median, ref)
    X = np.where(np.isnan(X), median, X)
    low, span = ref.min(axis=0), np.ptp(ref, axis=0)
    return (X - low) / np.where(span > 0, span, 1.0)
