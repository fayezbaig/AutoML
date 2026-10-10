"""Verify search_walltime_seconds is genuinely respected -- not just
configured. Uses a deliberately tiny budget and confirms fit() completes
promptly rather than running unboundedly."""
import time
import numpy as np
import pandas as pd
from automl.core import AutoML


def make_synthetic(n_rows, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_features)),
                     columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.normal(size=n_rows))
    return X, y


X, y = make_synthetic(2000)

TINY_BUDGET = 15  # seconds per algorithm -- should force near-immediate stops
model = AutoML(seed=0, portfolio=("lightgbm", "random_forest"),
               search_walltime_seconds=TINY_BUDGET, use_tabpfn=False)

t0 = time.time()
model.fit(X, y)
elapsed = time.time() - t0

n_algos = len(model.portfolio)
# generous 2x safety margin for overhead
max_expected = n_algos * TINY_BUDGET * 2

print(f"Configured budget: {TINY_BUDGET}s x {n_algos} algorithms")
print(f"Actual elapsed time: {elapsed:.1f}s")
print(f"Max expected (with overhead margin): {max_expected}s")

assert elapsed < max_expected, (
    f"Walltime limit NOT respected: took {elapsed:.1f}s, expected under {max_expected}s. "
    f"This means our 24h submission safety margin is NOT actually enforced!"
)
print("PASS: walltime_limit genuinely bounds search duration")
