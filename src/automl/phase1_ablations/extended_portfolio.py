"""
extended_portfolio.py — adds XGBoost, CatBoost, ElasticNet to the
comparison, each with the CORRECT preprocessing for that model:

  - XGBoost:    same label-encoding as LightGBM/RandomForest
  - CatBoost:   RAW categorical columns (its whole selling point --
                do NOT encode_categoricals first)
  - ElasticNet: label-encoding + StandardScaler (linear models are
                sensitive to feature scale, trees are not)

Self-contained (doesn't modify optimizer_comparison.py) to avoid
breaking the existing, working SMAC/NePS/Optuna comparison code.
Uses random search only (cheap baseline) -- swap in run_smac_bohb-style
search later if these prove competitive enough to invest further.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from automl.preprocessing import preprocess_for_algo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

from automl.data import Dataset
from automl.optimizer_comparison import DATADIR, encode_categoricals


def build_xgboost(config, seed=0):
    import xgboost as xgb
    return xgb.XGBRegressor(random_state=seed, verbosity=0, n_jobs=1, **config)


def build_catboost(config, seed=0, cat_features=None):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                             cat_features=cat_features, **config)


def build_elasticnet(config, seed=0):
    return ElasticNet(random_state=seed, **config)


SEARCH_BOUNDS = {
    "xgboost": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 1e-3, 0.3),
        "min_child_weight": ("int", 1, 20),
    },
    "catboost": {
        "iterations": ("int", 50, 300),
        "depth": ("int", 3, 10),
        "learning_rate": ("log_float", 1e-3, 0.3),
        "l2_leaf_reg": ("log_float", 1e-2, 10.0),
    },
    "elasticnet": {
        "alpha": ("log_float", 1e-4, 10.0),
        "l1_ratio": ("float", 0.0, 1.0),
    },
}


def sample_random_config(bounds, rng):
    config = {}
    for name, spec in bounds.items():
        kind, lo, hi = spec
        if kind == "int":
            config[name] = int(rng.integers(lo, hi + 1))
        elif kind == "float":
            config[name] = float(rng.uniform(lo, hi))
        elif kind == "log_float":
            config[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return config


def build_model(algo_name, config, seed=0, cat_features=None):
    if algo_name == "catboost":
        return build_catboost(config, seed=seed, cat_features=cat_features)
    elif algo_name == "xgboost":
        return build_xgboost(config, seed=seed)
    elif algo_name == "elasticnet":
        return build_elasticnet(config, seed=seed)
    raise ValueError(f"Unknown algorithm: {algo_name}")


def train_and_score(algo_name, config, X_train, y_train, X_val, y_val, seed=0, cat_features=None):
    model = build_model(algo_name, config, seed=seed, cat_features=cat_features)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return r2_score(y_val, preds)


def internal_cv_score(algo_name, config, X_train, y_train, seed, n_splits=5, cat_features=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    is_df = hasattr(X_train, "iloc")

    if is_df:
        X_reset = X_train.reset_index(drop=True)
        y_reset = y_train.reset_index(drop=True)
    else:
        X_reset = X_train
        y_reset = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else np.asarray(y_train)

    scores = []
    for tr_idx, val_idx in kf.split(X_reset):
        if is_df:
            X_tr, X_v = X_reset.iloc[tr_idx], X_reset.iloc[val_idx]
        else:
            X_tr, X_v = X_reset[tr_idx], X_reset[val_idx]

        if hasattr(y_reset, "iloc"):
            y_tr, y_v = y_reset.iloc[tr_idx], y_reset.iloc[val_idx]
        else:
            y_tr, y_v = y_reset[tr_idx], y_reset[val_idx]

        scores.append(train_and_score(algo_name, config, X_tr, y_tr, X_v, y_v, seed, cat_features))
    return float(np.mean(scores))


def run_random_search(algo_name, X_train, y_train, seed, budget, cat_features=None):
    rng = np.random.default_rng(seed)
    bounds = SEARCH_BOUNDS[algo_name]
    best_config, best_score = None, -np.inf
    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        score = internal_cv_score(algo_name, config, X_train, y_train, seed, cat_features=cat_features)
        if score > best_score:
            best_score, best_config = score, config
    return best_config


def compare_extended(algo_name, task, folds=None, budget=30, seed=0):
    folds = folds if folds is not None else list(range(1, 11))
    test_scores = []

    for fold in folds:
        ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
        X_train, X_test, cat_features = preprocess_for_algo(algo_name, ds.X_train, ds.X_test)
        y_train, y_test = ds.y_train, ds.y_test

        best_config = run_random_search(algo_name, X_train, y_train, seed, budget, cat_features)
        model = build_model(algo_name, best_config, seed=seed, cat_features=cat_features)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_scores.append(r2_score(y_test, preds))

    test_scores = np.array(test_scores)
    print(f"[{algo_name}/{task}] Random search TEST R2: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    return test_scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["xgboost", "catboost", "elasticnet"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scores = compare_extended(args.algo, args.task, budget=args.budget, seed=args.seed)

    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    np.save(f"results/{args.algo}_{args.task}_rs_test_b{args.budget}_seed{args.seed}.npy", scores)
    print(f"Saved to results/{args.algo}_{args.task}_rs_test_b{args.budget}_seed{args.seed}.npy")
