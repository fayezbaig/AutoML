"""
joint_cash.py — TRUE joint CASH (Combined Algorithm Selection and
Hyperparameter optimization): ONE ConfigurationSpace where algorithm
choice is a top-level categorical hyperparameter, with per-algorithm
hyperparameters conditional on that choice. A single SMAC-BOHB search
with ONE shared surrogate/run-history explores algorithm choice and
hyperparameters jointly, letting Hyperband abandon a weak algorithm
branch early and reallocate its budget -- unlike per-algorithm
independent searches (optimizer_comparison.py's run_smac_bohb /
catboost_bohb.py's run_catboost_bohb), which cannot share information
or budget across algorithms.

Fidelity is unified across algorithms as "n_estimators_budget", mapped
internally to each algorithm's actual round-count parameter
(n_estimators for LightGBM/RandomForest, iterations for CatBoost).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from automl.optimizer_comparison import encode_categoricals
from automl.extended_portfolio import preprocess_for_algo


def build_joint_configspace(seed=0):
    from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float
    from ConfigSpace import EqualsCondition

    cs = ConfigurationSpace(seed=seed)

    algorithm = Categorical("algorithm", ["lightgbm", "random_forest", "catboost"])
    cs.add([algorithm])

    # LightGBM-specific hyperparameters
    lgb_lr = Float("lgb_learning_rate", (1e-3, 0.3), log=True)
    lgb_leaves = Integer("lgb_num_leaves", (8, 256))
    lgb_depth = Integer("lgb_max_depth", (3, 12))
    lgb_min_child = Integer("lgb_min_child_samples", (5, 100))
    cs.add([lgb_lr, lgb_leaves, lgb_depth, lgb_min_child])
    cs.add([
        EqualsCondition(lgb_lr, algorithm, "lightgbm"),
        EqualsCondition(lgb_leaves, algorithm, "lightgbm"),
        EqualsCondition(lgb_depth, algorithm, "lightgbm"),
        EqualsCondition(lgb_min_child, algorithm, "lightgbm"),
    ])

    # RandomForest-specific hyperparameters
    rf_depth = Integer("rf_max_depth", (2, 32))
    rf_min_leaf = Integer("rf_min_samples_leaf", (1, 20))
    cs.add([rf_depth, rf_min_leaf])
    cs.add([
        EqualsCondition(rf_depth, algorithm, "random_forest"),
        EqualsCondition(rf_min_leaf, algorithm, "random_forest"),
    ])

    # CatBoost-specific hyperparameters
    cb_lr = Float("cb_learning_rate", (1e-3, 0.3), log=True)
    cb_depth = Integer("cb_depth", (3, 10))
    cb_l2 = Float("cb_l2_leaf_reg", (1e-2, 10.0), log=True)
    cs.add([cb_lr, cb_depth, cb_l2])
    cs.add([
        EqualsCondition(cb_lr, algorithm, "catboost"),
        EqualsCondition(cb_depth, algorithm, "catboost"),
        EqualsCondition(cb_l2, algorithm, "catboost"),
    ])

    return cs


def _config_to_model_kwargs(config, n_estimators_budget):
    """Translate a sampled joint config into (algo_name, kwargs dict)
    for that specific algorithm, injecting the shared fidelity value
    under whatever name that algorithm actually uses for round count."""
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


def _joint_cv_score(algo, kwargs, X_train_enc, X_train_cb, cat_features, y_train, seed, n_splits=5):
    """Internal K-fold CV objective, dispatching to the correct
    preprocessing (raw categoricals for CatBoost, label-encoded
    otherwise) based on which algorithm this trial sampled."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if algo == "catboost":
        X_reset = X_train_cb.reset_index(drop=True)
    else:
        X_reset = X_train_enc.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    scores = []
    for tr_idx, val_idx in kf.split(X_reset):
        X_tr, X_v = X_reset.iloc[tr_idx], X_reset.iloc[val_idx]
        y_tr, y_v = y_reset.iloc[tr_idx], y_reset.iloc[val_idx]

        if algo == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMRegressor(random_state=seed, verbosity=-1, num_threads=1,
                                      n_jobs=1, **kwargs)
        elif algo == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=seed, **kwargs)
        elif algo == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                      cat_features=cat_features, **kwargs)

        model.fit(X_tr, y_tr)
        preds = model.predict(X_v)
        scores.append(r2_score(y_v, preds))

    return float(np.mean(scores))


def run_joint_cash_bohb(X_train_raw, y_train, seed, root_dir,
                        min_budget=60, max_budget=300, n_trials=300, eta=3, n_init=15,
                        walltime_limit=None):
    """
    ONE joint search over algorithm choice + conditional hyperparameters,
    with a single shared SMAC-BOHB run-history/surrogate across all three
    algorithms. Returns (algo_name, kwargs) for the winning incumbent.
    """
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI

    X_train_enc = encode_categoricals(X_train_raw)
    X_train_cb, _, cat_features = preprocess_for_algo("catboost", X_train_raw, X_train_raw)

    cs = build_joint_configspace(seed)

    def train(config, seed: int = 0, budget: float = max_budget):
        algo, kwargs = _config_to_model_kwargs(config, budget)
        score = _joint_cv_score(algo, kwargs, X_train_enc, X_train_cb, cat_features, y_train, seed)
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
    acquisition_function = EI()
    model = MultiFidelityFacade.get_model(scenario, n_trees=200)
    intensifier = Hyperband(scenario, eta=eta, incumbent_selection="highest_budget")

    smac = MultiFidelityFacade(
        scenario, train, model=model, initial_design=initial_design,
        acquisition_function=acquisition_function, intensifier=intensifier,
        overwrite=True,
    )
    incumbent = smac.optimize()

    algo, kwargs = _config_to_model_kwargs(dict(incumbent), max_budget)
    return algo, kwargs


def refit_joint(algo, kwargs, X_train_raw, y_train, seed):
    """Refit the winning (algorithm, kwargs) pair on full data, with the
    correct preprocessing for that specific algorithm."""
    if algo == "catboost":
        from catboost import CatBoostRegressor
        X_train_cb, _, cat_features = preprocess_for_algo("catboost", X_train_raw, X_train_raw)
        model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                  cat_features=cat_features, **kwargs)
        model.fit(X_train_cb, y_train)
        return model, (lambda m, X: m.predict(preprocess_for_algo("catboost", X, X)[0]))
    else:
        X_train_enc = encode_categoricals(X_train_raw)
        if algo == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMRegressor(random_state=seed, verbosity=-1, num_threads=1,
                                      n_jobs=1, **kwargs)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=seed, **kwargs)
        model.fit(X_train_enc, y_train)
        return model, (lambda m, X: m.predict(encode_categoricals(X)))


if __name__ == "__main__":
    import argparse
    from automl.data import Dataset
    from automl.optimizer_comparison import DATADIR
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--walltime", type=float, default=None,
                        help="Hard wall-clock cap in seconds for the joint search.")
    args = parser.parse_args()

    ds = Dataset.load(datadir=DATADIR, task=args.task, fold=args.fold)
    X_train, X_val, y_train, y_val = train_test_split(
        ds.X_train, ds.y_train, test_size=0.2, random_state=args.seed
    )

    algo, kwargs = run_joint_cash_bohb(
        X_train, y_train, args.seed,
        root_dir=f"results/joint_cash_{args.task}",
        walltime_limit=args.walltime,
    )
    print(f"Joint CASH winner: algorithm={algo}, config={kwargs}")

    model, predict_fn = refit_joint(algo, kwargs, X_train, y_train, args.seed)
    val_preds = predict_fn(model, X_val)
    score = r2_score(y_val, val_preds)
    print(f"[{args.task}] Joint CASH-BOHB val R2: {score:.4f} (algorithm={algo})")
