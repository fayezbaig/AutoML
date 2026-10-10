"""
hyperparam_importance.py — approximate hyperparameter importance analysis,
as taught in Assignment 8 / week 12 (Analysis: fANOVA, LPI).

Fits a RandomForest surrogate on (config -> score) pairs from a fresh
random search, reads off feature_importances_ as an approximation of
fANOVA-style importance. Supports lightgbm/random_forest (via
optimizer_comparison) and catboost (via extended_portfolio, with its
own correct raw-categorical preprocessing).
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from automl.data import Dataset
from automl.optimizer_comparison import (
    DATADIR, encode_categoricals, ALGORITHMS as LGB_RF_ALGORITHMS,
    sample_random_config as lgb_rf_sample_config,
    internal_cv_score as lgb_rf_internal_cv_score,
)
from automl.extended_portfolio import (
    preprocess_for_algo, SEARCH_BOUNDS as EXT_SEARCH_BOUNDS,
    sample_random_config as ext_sample_config,
    internal_cv_score as ext_internal_cv_score,
)


def collect_config_score_pairs(algo_name, task, seed=0, n_samples=50, fold=1):
    ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)

    if algo_name == "catboost":
        X_train, _, cat_features = preprocess_for_algo("catboost", ds.X_train, ds.X_test)
        bounds = EXT_SEARCH_BOUNDS["catboost"]
        rng = np.random.default_rng(seed)
        param_names = list(bounds.keys())
        configs, scores = [], []
        for _ in range(n_samples):
            config = ext_sample_config(bounds, rng)
            score = ext_internal_cv_score("catboost", config, X_train, ds.y_train, seed,
                                          cat_features=cat_features)
            configs.append([config[p] for p in param_names])
            scores.append(score)
        return np.array(configs), np.array(scores), param_names

    else:
        X_train = encode_categoricals(ds.X_train)
        bounds = LGB_RF_ALGORITHMS[algo_name]["random_bounds"]
        rng = np.random.default_rng(seed)
        param_names = list(bounds.keys())
        configs, scores = [], []
        for _ in range(n_samples):
            config = lgb_rf_sample_config(bounds, rng)
            score = lgb_rf_internal_cv_score(algo_name, config, X_train, ds.y_train, seed)
            configs.append([config[p] for p in param_names])
            scores.append(score)
        return np.array(configs), np.array(scores), param_names


def compute_importance(configs, scores, param_names, seed=0):
    surrogate = RandomForestRegressor(n_estimators=100, random_state=seed)
    surrogate.fit(configs, scores)
    importances = surrogate.feature_importances_
    return sorted(zip(param_names, importances), key=lambda x: -x[1])


def run_importance_analysis(algo_name, task, seed=0, n_samples=50):
    print(f"=== {algo_name}/{task}: collecting {n_samples} (config, score) samples ===")
    configs, scores, param_names = collect_config_score_pairs(algo_name, task, seed, n_samples)

    print(f"Score range observed: {scores.min():.4f} to {scores.max():.4f}")

    ranked = compute_importance(configs, scores, param_names, seed)

    print(f"\n=== Approximate hyperparameter importance ({algo_name}/{task}) ===")
    for name, importance in ranked:
        bar = "#" * int(importance * 50)
        print(f"{name:20s} {importance:.4f}  {bar}")

    return ranked


if __name__ == "__main__":
    from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
    reexec_with_fixed_hash_seed()
    pin_thread_env()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["lightgbm", "random_forest", "catboost"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_importance_analysis(args.algo, args.task, seed=args.seed, n_samples=args.n_samples)
