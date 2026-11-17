import numpy as np
import pytest

from biomaterial_optimizer.models.bo import expected_improvement, run, scale


def smooth_pool(n=60, seed=0):
    """Fake recipes on a smooth hill: the best one is near x = 0.7."""
    x = np.random.default_rng(seed).uniform(0, 1, (n, 1))
    return x, -((x[:, 0] - 0.7) ** 2)


def test_expected_improvement_rewards_good_and_uncertain():
    ei = expected_improvement(
        mean=np.array([1.0, 0.0, 0.0]), std=np.array([0.1, 0.1, 1.0]), best=0.5
    )
    assert ei[0] > ei[1]  # predicted better than the best so far
    assert ei[2] > ei[1]  # same prediction, but more uncertain


def test_run_never_picks_a_recipe_twice():
    X, y = smooth_pool(n=10)
    curve = run(X, y, "random", n_init=3, budget=20, rng=np.random.default_rng(0))
    assert len(curve) == 10  # stops when the pool is empty
    assert curve[-1] == y.max()


def test_curve_is_best_so_far():
    X, y = smooth_pool()
    curve = run(X, y, "random", n_init=5, budget=10, rng=np.random.default_rng(1))
    assert len(curve) == 15
    assert np.all(np.diff(curve) >= 0)


def test_bo_beats_random_on_a_smooth_problem():
    X, y = smooth_pool()
    top = np.quantile(y, 0.95)

    def rounds(strategy):
        found = []
        for seed in range(4):
            curve = run(X, y, strategy, 3, 15, np.random.default_rng(seed))
            hit = np.nonzero(curve[2:] >= top)[0]
            found.append(hit[0] if len(hit) else 16)
        return np.mean(found)

    assert rounds("bo") < rounds("random")


def test_bo_uses_any_surrogate():
    X, y = smooth_pool(n=20)
    calls = []

    def fake_gp(X_seen, y_seen, X_pool):
        calls.append(len(X_seen))
        return X_pool[:, 0], np.ones(len(X_pool))  # "bigger x is better"

    curve = run(X, y, "bo", 3, 2, np.random.default_rng(0), surrogate=fake_gp)
    assert calls == [3, 4]
    assert len(curve) == 5


def test_scale():
    X = np.array([[0.0, 5.0, np.nan], [10.0, 5.0, 2.0], [5.0, 5.0, 4.0]])
    scaled = scale(X)
    assert scaled[:, 0] == pytest.approx([0, 1, 0.5])
    assert scaled[:, 1] == pytest.approx([0, 0, 0])  # never varies
    assert scaled[0, 2] == pytest.approx(0.5)  # gap -> median (3) -> middle
