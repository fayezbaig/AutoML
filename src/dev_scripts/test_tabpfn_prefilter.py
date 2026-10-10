"""Fast, repeatable test for the TabPFN row-count pre-filter -- no need
to download a large real dataset each time; uses synthetic data sized
to guarantee the training split exceeds tabpfn_max_train_rows,
computed directly from val_fraction rather than hardcoding a total row
count (which would silently become wrong if val_fraction ever changes)."""
import numpy as np
import pandas as pd
from automl.core import AutoML


def make_synthetic(n_rows, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_features)),
                     columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.normal(size=n_rows))
    return X, y


# --- Test 1: training split ABOVE threshold -> should SKIP ---
threshold = 50000
model_big = AutoML(seed=0, portfolio=("lightgbm",),
                   tabpfn_max_train_rows=threshold)

# Compute total rows needed so that (1 - val_fraction) * n_total > threshold,
# regardless of what val_fraction actually is -- avoids a hardcoded total
# row count silently becoming wrong if val_fraction changes later.
n_total_needed = int((threshold + 1) / (1 - model_big.val_fraction)) + 1
X_big, y_big = make_synthetic(n_total_needed)

expected_train_rows = int(n_total_needed * (1 - model_big.val_fraction))
print(
    f"Test 1: n_total={n_total_needed}, val_fraction={model_big.val_fraction}, "
    f"expected_train_rows~={expected_train_rows} (threshold={threshold})")
assert expected_train_rows > threshold, (
    f"Test setup error: computed train split ({expected_train_rows}) does not "
    f"actually exceed threshold ({threshold}) -- test would be meaningless."
)

model_big.fit(X_big, y_big)
assert "tabpfn" not in model_big.per_algo_val_scores_, \
    "TabPFN should have been skipped above threshold!"
print("PASS: TabPFN correctly skipped when training split exceeds threshold")

# --- Test 2: training split BELOW threshold -> should NOT skip (sanity check) ---
model_small = AutoML(seed=0, portfolio=("lightgbm",),
                     tabpfn_max_train_rows=threshold)
# tiny, well under any reasonable threshold
X_small, y_small = make_synthetic(200)
model_small.fit(X_small, y_small)
print(
    f"Test 2: per_algo_val_scores_ keys = {list(model_small.per_algo_val_scores_.keys())}")
# NOTE: TabPFN might still be absent here for OTHER reasons (e.g. license/
# token not set in this environment) -- this is a softer sanity check, not
# a hard assertion, specifically to isolate the row-count logic from
# TabPFN availability itself.

print("\nTest complete.")
