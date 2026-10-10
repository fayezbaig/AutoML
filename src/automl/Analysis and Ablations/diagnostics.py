"""
diagnostics.py — one-off verification and sanity-check tools, merged
from budget_estimator.py, significance_test.py, and test_large_dataset.py.
Not part of the production pipeline (automl.py) -- these are development-
time checks: how expensive is a search likely to be, is a result
statistically significant, does the pipeline generalize to unfamiliar
data at a different scale.

Usage:
    python -m automl.diagnostics budget --task wine_quality
    python -m automl.diagnostics significance --algo lightgbm --task wine_quality --mode smac_bohb
    python -m automl.diagnostics large_dataset
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import argparse


# ============================================================
# BUDGET ESTIMATION (from budget_estimator.py)
# ============================================================

def measure_avg_cost(algo_name, X_train, y_train, X_val, y_val, n_probe=5, seed=0):
    """Run a handful of real evaluations with randomly sampled configs
    to estimate average wall-clock cost per evaluation."""
    from automl.search_methods import train_and_score
    from automl.search_spaces import RANDOM_BOUNDS, sample_random_config

    rng = np.random.default_rng(seed)
    bounds = RANDOM_BOUNDS[algo_name]
    times = []
    for _ in range(n_probe):
        config = sample_random_config(bounds, rng)
        t0 = time.time()
        train_and_score(algo_name, config, X_train, y_train, X_val, y_val)
        times.append(time.time() - t0)
    return float(np.mean(times)), float(np.std(times))


def budget_for_time_allocation(algo_name, X_train, y_train, X_val, y_val,
                               allocated_seconds, safety_margin=0.8, n_probe=5):
    """Convert a time allocation (seconds) into a safe max_evaluations_total,
    based on measured per-eval cost."""
    avg_cost, std_cost = measure_avg_cost(algo_name, X_train, y_train, X_val, y_val, n_probe=n_probe)
    usable_seconds = allocated_seconds * safety_margin
    n_evals = max(1, int(usable_seconds / avg_cost))
    print(
        f"[{algo_name}] measured {avg_cost:.3f}s ± {std_cost:.3f}s per eval "
        f"(from {n_probe} probes) -> budget={n_evals} evals "
        f"for a {allocated_seconds}s allocation ({int((1-safety_margin)*100)}% safety margin)"
    )
    return n_evals


def run_budget_diagnostic(task, fold=1):
    from automl.data import Dataset
    from automl.search_methods import DATADIR
    from sklearn.model_selection import train_test_split

    ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
    X_train, X_val, y_train, y_val = train_test_split(
        ds.X_train, ds.y_train, test_size=0.2, random_state=0
    )

    ALLOCATIONS = {"lightgbm": 2 * 3600, "random_forest": 1 * 3600}
    print(f"Estimating budgets for task={task}, fold={fold}\n")
    for algo_name, seconds in ALLOCATIONS.items():
        budget_for_time_allocation(algo_name, X_train, y_train, X_val, y_val, seconds)


# ============================================================
# SIGNIFICANCE TESTING (from significance_test.py)
# ============================================================

def run_significance_diagnostic(algo, task, mode, budget=30, seed=0):
    from scipy.stats import ttest_rel

    rs = np.load(f"results/{algo}_{task}_rs_test_b{budget}_seed{seed}.npy")
    method = np.load(f"results/{algo}_{task}_{mode}_test_b{budget}_seed{seed}.npy")

    t_stat, p_value = ttest_rel(method, rs)
    sig = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "not significant"
    print(f"[{task}] {mode} vs Random Search: t={t_stat:.3f}, p={p_value:.3f} -> {sig}")


# ============================================================
# LARGE-DATASET STRESS TEST (from test_large_dataset.py)
# ============================================================

def fetch_diamonds():
    """OpenML dataset id 42225 -- diamonds, ~54k rows, price regression."""
    import openml
    dataset = openml.datasets.get_dataset(42225)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    print(f"Fetched '{dataset.name}': shape={X.shape}, target={dataset.default_target_attribute}")
    return X, y


def quick_model_test(algo_name, X_train, y_train, X_val, y_val, cat_features=None, seed=0):
    from sklearn.metrics import r2_score

    if algo_name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                  cat_features=cat_features, iterations=200)
    elif algo_name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=1, n_estimators=200)
    elif algo_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=seed, n_estimators=200)
    elif algo_name == "tabpfn":
        from tabpfn import TabPFNRegressor
        model = TabPFNRegressor()
    else:
        raise ValueError(algo_name)

    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    preds = model.predict(X_val)
    predict_time = time.time() - t0

    score = r2_score(y_val, preds)
    print(f"[{algo_name}] R2={score:.4f}  fit_time={fit_time:.2f}s  predict_time={predict_time:.2f}s "
          f"(n_train={len(X_train)}, n_val={len(X_val)})")
    return score, fit_time, predict_time


def run_large_dataset_diagnostic():
    from sklearn.model_selection import train_test_split
    from automl.preprocessing import encode_categoricals, preprocess_for_algo

    X, y = fetch_diamonds()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}\n")

    X_train_enc = encode_categoricals(X_train)
    X_val_enc = encode_categoricals(X_val)
    for algo in ["lightgbm", "random_forest"]:
        quick_model_test(algo, X_train_enc, y_train, X_val_enc, y_val)

    X_train_cb, X_val_cb, cat_features = preprocess_for_algo("catboost", X_train, X_val)
    quick_model_test("catboost", X_train_cb, y_train, X_val_cb, y_val, cat_features=cat_features)

    try:
        quick_model_test("tabpfn", X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"[tabpfn] FAILED: {e}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Development diagnostics -- not part of the production pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_budget = subparsers.add_parser("budget", help="Estimate per-evaluation cost and safe trial budgets.")
    p_budget.add_argument("--task", required=True)
    p_budget.add_argument("--fold", type=int, default=1)

    p_sig = subparsers.add_parser("significance", help="Paired t-test: search method vs. random search.")
    p_sig.add_argument("--algo", required=True)
    p_sig.add_argument("--task", required=True)
    p_sig.add_argument("--mode", required=True)
    p_sig.add_argument("--budget", type=int, default=30)
    p_sig.add_argument("--seed", type=int, default=0)

    p_large = subparsers.add_parser("large_dataset", help="Stress-test on a larger out-of-sample OpenML dataset.")

    args = parser.parse_args()

    if args.command == "budget":
        run_budget_diagnostic(args.task, fold=args.fold)
    elif args.command == "significance":
        run_significance_diagnostic(args.algo, args.task, args.mode, budget=args.budget, seed=args.seed)
    elif args.command == "large_dataset":
        run_large_dataset_diagnostic()
