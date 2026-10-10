"""
search_methods.py — PRODUCTION search methods, merged from
optimizer_comparison.py (SMAC variants), catboost_bohb.py, and
joint_cash.py. This is what automl.py actually calls.

PRIMARY / RECOMMENDED: run_joint_cash_bohb() -- a single joint CASH
search (algorithm choice + conditional hyperparameters in ONE
ConfigurationSpace, ONE shared SMAC-BOHB surrogate/run-history). This
lets Hyperband abandon a weak algorithm branch early and reallocate its
budget -- unlike independent per-algorithm searches. Validated on
wine_quality: converges to the correct winning algorithm (CatBoost) and
matches/exceeds independent per-algorithm search given enough budget
(300 trials: joint=0.4975 vs independent CatBoost-only=0.4958).

RETAINED for comparison-table reproducibility and as algorithm-specific
fallbacks: run_smac_bo, run_smac_mf_datafrac, run_smac_mf_rounds,
run_smac_bohb (per-algorithm), run_catboost_bohb (CatBoost-only).

See phase1_explorations.py for NePS/Optuna/XGBoost/ElasticNet/meta-
learning -- tested during development, not part of this production path.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import logging
# Silence noisy library internals (per-trial SMAC logs, etc.) while
# keeping our own automl.py logger.info() progress lines visible.
for _noisy_logger in ("smac", "ConfigSpace", "lightgbm"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

from automl.data import Dataset
from automl.preprocessing import encode_categoricals, preprocess_for_algo
from automl.search_spaces import (
    RANDOM_BOUNDS, sample_random_config, build_configspace, build_joint_configspace,
)

FILE = Path(__file__).resolve()
DATADIR = FILE.parents[2] / "data"
TASK = "yprop_4_1"
BUDGET = 30
N_SPLITS = 5


# ============================================================
# Model builders (single-algorithm)
# ============================================================

def build_lgbm(config, seed=0):
    import lightgbm as lgb
    return lgb.LGBMRegressor(random_state=seed, verbosity=-1, num_threads=1, n_jobs=1, **config)


def build_rf(config, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=seed, **config)


def build_catboost(config, seed=0, cat_features=None):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                             cat_features=cat_features, **config)


ALGORITHMS = {
    "lightgbm": {"builder": build_lgbm},
    "random_forest": {"builder": build_rf},
}


def train_and_score(algo_name, config, X_train, y_train, X_val, y_val, seed=0, cat_features=None):
    if algo_name == "catboost":
        model = build_catboost(config, seed=seed, cat_features=cat_features)
    else:
        model = ALGORITHMS[algo_name]["builder"](config, seed=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return r2_score(y_val, preds)


def internal_cv_score(algo_name, config, X_train, y_train, seed, budget=None, n_splits=N_SPLITS,
                      cat_features=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    cfg = dict(config)
    if budget is not None:
        budget_key = "iterations" if algo_name == "catboost" else "n_estimators"
        cfg[budget_key] = int(budget)

    scores = []
    for tr_idx, val_idx in kf.split(X_reset):
        X_tr, X_v = X_reset.iloc[tr_idx], X_reset.iloc[val_idx]
        y_tr, y_v = y_reset.iloc[tr_idx], y_reset.iloc[val_idx]
        scores.append(train_and_score(algo_name, cfg, X_tr, y_tr, X_v, y_v, seed, cat_features))
    return float(np.mean(scores))


def refit_final(algo_name, config, X_train, y_train, seed, use_early_stopping=False,
                cat_features=None, val_fraction=0.15, max_estimators=2000,
                early_stopping_rounds=50):
    if algo_name == "lightgbm" and use_early_stopping:
        import lightgbm as lgb
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction, random_state=seed
        )
        probe = lgb.LGBMRegressor(random_state=seed, n_estimators=max_estimators,
                                  verbosity=-1, num_threads=1, n_jobs=1, **config)
        probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        n_rounds = probe.best_iteration_ or max_estimators
        final_model = lgb.LGBMRegressor(random_state=seed, n_estimators=n_rounds,
                                        verbosity=-1, num_threads=1, n_jobs=1, **config)
        final_model.fit(X_train, y_train)
        return final_model

    if algo_name == "catboost":
        model = build_catboost(config, seed=seed, cat_features=cat_features)
    else:
        model = ALGORITHMS[algo_name]["builder"](config, seed=seed)
    model.fit(X_train, y_train)
    return model


def run_random_search(algo_name, X_train, y_train, seed, budget, cat_features=None):
    rng = np.random.default_rng(seed)
    bounds = RANDOM_BOUNDS[algo_name]
    best_config, best_score = None, -np.inf
    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        score = internal_cv_score(algo_name, config, X_train, y_train, seed, cat_features=cat_features)
        if score > best_score:
            best_score, best_config = score, config
    return best_config


# ============================================================
# Per-algorithm SMAC searches (retained for comparison tables)
# ============================================================

def run_smac_bo(algo_name, X_train, y_train, seed, budget, root_dir, walltime_limit=None):
    from smac import HyperparameterOptimizationFacade, Scenario
    cs = build_configspace(algo_name, seed)

    def train(config, seed: int = 0):
        return -internal_cv_score(algo_name, dict(config), X_train, y_train, seed)

    scenario_kwargs = dict(n_trials=budget, seed=seed, output_directory=root_dir, deterministic=True)
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)
    smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
    incumbent = smac.optimize()
    return dict(incumbent)


def run_smac_mf_datafrac(algo_name, X_train, y_train, seed, budget, root_dir, walltime_limit=None):
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    cs = build_configspace(algo_name, seed)

    def train(config, seed: int = 0, budget: float = 1.0):
        n = max(50, int(len(X_train) * budget))
        X_sub, y_sub = X_train.iloc[:n], y_train.iloc[:n]
        return -internal_cv_score(algo_name, dict(config), X_sub, y_sub, seed)

    scenario_kwargs = dict(n_trials=budget, seed=seed, output_directory=root_dir,
                           min_budget=0.1, max_budget=1.0, deterministic=True)
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)
    smac = MultiFidelityFacade(scenario, train,
                               intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
                               overwrite=True)
    incumbent = smac.optimize()
    return dict(incumbent)


def run_smac_mf_rounds(algo_name, X_train, y_train, seed, budget, root_dir,
                       min_budget=60, max_budget=300, walltime_limit=None):
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    cs = build_configspace(algo_name, seed, exclude_fidelity=True)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score(algo_name, dict(config), X_train, y_train, seed, budget=budget)

    scenario_kwargs = dict(n_trials=budget, seed=seed, output_directory=root_dir,
                           min_budget=min_budget, max_budget=max_budget, deterministic=True)
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)
    smac = MultiFidelityFacade(scenario, train,
                               intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
                               overwrite=True)
    incumbent = smac.optimize()
    return dict(incumbent)


def run_smac_bohb(algo_name, X_train, y_train, seed, root_dir,
                  min_budget=60, max_budget=300, n_trials=60, eta=3, n_init=10,
                  walltime_limit=None):
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI
    cs = build_configspace(algo_name, seed, exclude_fidelity=True)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score(algo_name, dict(config), X_train, y_train, seed, budget=budget)

    scenario_kwargs = dict(name="bohb", output_directory=root_dir, n_trials=n_trials,
                           deterministic=True, seed=seed, min_budget=min_budget,
                           max_budget=max_budget, n_workers=1)
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=n_init)
    model = MultiFidelityFacade.get_model(scenario, n_trees=200)
    intensifier = Hyperband(scenario, eta=eta, incumbent_selection="highest_budget")
    smac = MultiFidelityFacade(scenario, train, model=model, initial_design=initial_design,
                               acquisition_function=EI(), intensifier=intensifier, overwrite=True)
    incumbent = smac.optimize()
    return dict(incumbent)


def run_catboost_bohb(X_train, y_train, cat_features, seed, root_dir,
                      min_budget=60, max_budget=300, n_trials=60, eta=3, n_init=10,
                      walltime_limit=None):
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI
    cs = build_configspace("catboost", seed)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score("catboost", dict(config), X_train, y_train, seed,
                                  budget=budget, cat_features=cat_features)

    scenario_kwargs = dict(name="catboost_bohb", output_directory=root_dir, n_trials=n_trials,
                           deterministic=True, seed=seed, min_budget=min_budget,
                           max_budget=max_budget, n_workers=1)
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=n_init)
    model = MultiFidelityFacade.get_model(scenario, n_trees=200)
    intensifier = Hyperband(scenario, eta=eta, incumbent_selection="highest_budget")
    smac = MultiFidelityFacade(scenario, train, model=model, initial_design=initial_design,
                               acquisition_function=EI(), intensifier=intensifier, overwrite=True)
    incumbent = smac.optimize()
    if incumbent is None:
        # Walltime exhausted before any trial completed (e.g. system
        # slept mid-search) -- fall back to safe defaults rather than
        # crash, so a single bad fold doesn't kill a multi-fold run.
        print("WARNING: no incumbent found (walltime exhausted before "
              "any trial completed) -- falling back to default config.")
        best_config = {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0}
    else:
        best_config = dict(incumbent)
    best_config["iterations"] = max_budget
    return best_config


# ============================================================
# JOINT CASH (PRIMARY / RECOMMENDED production search)
# ============================================================

def _config_to_model_kwargs(config, n_estimators_budget):
    algo = config["algorithm"]
    if algo == "lightgbm":
        kwargs = {
            "learning_rate": config["lgb_learning_rate"],
            "num_leaves": config["lgb_num_leaves"],
            "max_depth": config["lgb_max_depth"],
            "min_child_samples": config["lgb_min_child_samples"],
            "n_estimators": int(n_estimators_budget),
        }
    elif algo == "random_forest":
        kwargs = {
            "max_depth": config["rf_max_depth"],
            "min_samples_leaf": config["rf_min_samples_leaf"],
            "n_estimators": int(n_estimators_budget),
        }
    elif algo == "catboost":
        kwargs = {
            "learning_rate": config["cb_learning_rate"],
            "depth": config["cb_depth"],
            "l2_leaf_reg": config["cb_l2_leaf_reg"],
            "iterations": int(n_estimators_budget),
        }
    else:
        raise ValueError(algo)
    return algo, kwargs


def run_joint_cash_bohb(X_train_raw, y_train, seed, root_dir,
                        min_budget=60, max_budget=300, n_trials=300, eta=3, n_init=15,
                        walltime_limit=1800):
    """
    PRIMARY production search: ONE joint ConfigurationSpace (algorithm
    choice + conditional hyperparameters), ONE shared SMAC-BOHB
    surrogate/run-history. Validated (wine_quality): needs ~300 trials
    to reliably converge to the true best algorithm+config; at lower
    trial counts it can settle on a suboptimal algorithm (e.g. 90 trials
    -> RandomForest instead of the true-best CatBoost).

    walltime_limit defaults to 1800s (30 min) as a safety cap -- this is
    the exact mechanism that would have prevented the earlier
    superconductivity incident (an unbounded per-algorithm CatBoost
    search projected to ~12h before being manually killed).
    """
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI

    X_train_enc = encode_categoricals(X_train_raw)
    X_train_cb, _, cat_features = preprocess_for_algo("catboost", X_train_raw, X_train_raw)

    cs = build_joint_configspace(seed)

    def train(config, seed: int = 0, budget: float = max_budget):
        algo, kwargs = _config_to_model_kwargs(config, budget)
        if algo == "catboost":
            score = internal_cv_score(algo, kwargs, X_train_cb, y_train, seed, cat_features=cat_features)
        else:
            score = internal_cv_score(algo, kwargs, X_train_enc, y_train, seed)
        return -score

    scenario_kwargs = dict(
        name="joint_cash_bohb", output_directory=root_dir, n_trials=n_trials,
        deterministic=True, seed=seed, min_budget=min_budget, max_budget=max_budget,
        n_workers=1,
    )
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=n_init)
    model = MultiFidelityFacade.get_model(scenario, n_trees=200)
    intensifier = Hyperband(scenario, eta=eta, incumbent_selection="highest_budget")
    smac = MultiFidelityFacade(scenario, train, model=model, initial_design=initial_design,
                               acquisition_function=EI(), intensifier=intensifier, overwrite=True)
    incumbent = smac.optimize()

    algo, kwargs = _config_to_model_kwargs(dict(incumbent), max_budget)
    return algo, kwargs


def refit_joint(algo, kwargs, X_train_raw, y_train, seed):
    if algo == "catboost":
        X_train_cb, _, cat_features = preprocess_for_algo("catboost", X_train_raw, X_train_raw)
        model = build_catboost(kwargs, seed=seed, cat_features=cat_features)
        model.fit(X_train_cb, y_train)
        return model, (lambda m, X: m.predict(preprocess_for_algo("catboost", X, X)[0]))
    else:
        X_train_enc = encode_categoricals(X_train_raw)
        model = ALGORITHMS[algo]["builder"](kwargs, seed=seed)
        model.fit(X_train_enc, y_train)
        return model, (lambda m, X: m.predict(encode_categoricals(X)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_joint = subparsers.add_parser("joint_cash", help="Run the primary joint CASH-BOHB search.")
    p_joint.add_argument("--task", required=True)
    p_joint.add_argument("--fold", type=int, default=1)
    p_joint.add_argument("--seed", type=int, default=0)
    p_joint.add_argument("--n-trials", type=int, default=300)
    p_joint.add_argument("--walltime", type=float, default=1800)

    p_compare = subparsers.add_parser("compare", help="Per-algorithm search comparison (for tables).")
    p_compare.add_argument("--algo", required=True, choices=["lightgbm", "random_forest"])
    p_compare.add_argument("--task", default=TASK)
    p_compare.add_argument("--mode", choices=["smac_bo", "smac_mf_frac", "smac_mf_rounds", "smac_bohb"],
                           default="smac_bohb")
    p_compare.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "joint_cash":
        ds = Dataset.load(datadir=DATADIR, task=args.task, fold=args.fold)
        X_train, X_val, y_train, y_val = train_test_split(
            ds.X_train, ds.y_train, test_size=0.2, random_state=args.seed
        )
        algo, kwargs = run_joint_cash_bohb(
            X_train, y_train, args.seed, root_dir=f"results/joint_cash_{args.task}",
            n_trials=args.n_trials, walltime_limit=args.walltime,
        )
        print(f"Joint CASH winner: algorithm={algo}, config={kwargs}")
        model, predict_fn = refit_joint(algo, kwargs, X_train, y_train, args.seed)
        val_preds = predict_fn(model, X_val)
        score = r2_score(y_val, val_preds)
        print(f"[{args.task}] Joint CASH-BOHB val R2: {score:.4f} (algorithm={algo})")

    elif args.command == "compare":
        print("Use search_methods.run_smac_bohb / run_smac_bo etc. directly, "
              "or see caruana_ensemble.py for the full comparison pipeline.")
