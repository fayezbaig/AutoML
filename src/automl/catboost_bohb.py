"""
catboost_bohb.py — SMAC multi-fidelity (BOHB-style) tuning for CatBoost,
using RAW categorical features (its native strength) rather than
label-encoded ones. Fidelity = iterations (boosting rounds), matching
the same approach used for LightGBM elsewhere in this project.
"""
from automl.preprocessing import preprocess_for_algo
from automl.optimizer_comparison import DATADIR
from automl.data import Dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")


def internal_cv_score_catboost(
        config, X_train, y_train, cat_features, seed, budget=None, n_splits=5):
    from catboost import CatBoostRegressor
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    cfg = dict(config)
    if budget is not None:
        cfg["iterations"] = int(budget)

    scores = []
    for tr_idx, val_idx in kf.split(X_reset):
        X_tr, X_v = X_reset.iloc[tr_idx], X_reset.iloc[val_idx]
        y_tr, y_v = y_reset.iloc[tr_idx], y_reset.iloc[val_idx]

        model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                  cat_features=cat_features, **cfg)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_v)
        scores.append(r2_score(y_v, preds))
    return float(np.mean(scores))


def run_catboost_bohb(X_train, y_train, cat_features, seed, root_dir,
                      min_budget=60, max_budget=300, n_trials=60, eta=3, n_init=10,
                      walltime_limit=None):
    """walltime_limit (seconds): hard wall-clock cap -- see optimizer_comparison.py's
    run_smac_bohb for full rationale (this is the exact mechanism that would
    have prevented the superconductivity 12h-projection incident)."""
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI
    from automl.search_spaces import build_configspace
    cs = build_configspace("catboost", seed=seed)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score_catboost(dict(config), X_train, y_train, cat_features,
                                           seed, budget=budget)

    scenario_kwargs = dict(
        name="catboost_bohb", output_directory=root_dir, n_trials=n_trials,
        deterministic=True, seed=seed, min_budget=min_budget,
        max_budget=max_budget, n_workers=1,
    )
    if walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = walltime_limit
    scenario = Scenario(cs, **scenario_kwargs)

    initial_design = MultiFidelityFacade.get_initial_design(
        scenario, n_configs=n_init)
    acquisition_function = EI()
    model = MultiFidelityFacade.get_model(scenario, n_trees=200)
    intensifier = Hyperband(
        scenario,
        eta=eta,
        incumbent_selection="highest_budget")

    smac = MultiFidelityFacade(
        scenario, train, model=model, initial_design=initial_design,
        acquisition_function=acquisition_function, intensifier=intensifier,
        overwrite=True,
    )
    incumbent = smac.optimize()
    best_config = dict(incumbent)
    best_config["iterations"] = max_budget
    return best_config


def compare_catboost_bohb(task, folds=None, seed=0):
    folds = folds if folds is not None else list(range(1, 11))
    rs_scores, bohb_scores = [], []

    for fold in folds:
        ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
        X_train, X_test, cat_features = preprocess_for_algo(
            "catboost", ds.X_train, ds.X_test)
        y_train, y_test = ds.y_train, ds.y_test

        from automl.phase1_ablations.extended_portfolio import run_random_search as ep_random_search
        rs_config = ep_random_search(
            "catboost",
            X_train,
            y_train,
            seed,
            budget=30,
            cat_features=cat_features)
        from catboost import CatBoostRegressor
        rs_model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                     cat_features=cat_features, **rs_config)
        rs_model.fit(X_train, y_train)
        rs_scores.append(r2_score(y_test, rs_model.predict(X_test)))

        bohb_config = run_catboost_bohb(
            X_train, y_train, cat_features, seed,
            root_dir=f"results/catboost_bohb_{task}_fold{fold}",
        )
        bohb_model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                       cat_features=cat_features, **bohb_config)
        bohb_model.fit(X_train, y_train)
        bohb_scores.append(r2_score(y_test, bohb_model.predict(X_test)))

    rs_scores, bohb_scores = np.array(rs_scores), np.array(bohb_scores)
    print(
        f"[catboost/{task}] Random search TEST R2: {rs_scores.mean():.4f} ± {rs_scores.std():.4f}")
    print(
        f"[catboost/{task}] BOHB-tuned TEST R2: {bohb_scores.mean():.4f} ± {bohb_scores.std():.4f}")
    return rs_scores, bohb_scores


if __name__ == "__main__":
    from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
    reexec_with_fixed_hash_seed()
    pin_thread_env()
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rs_scores, bohb_scores = compare_catboost_bohb(args.task, seed=args.seed)

    Path("results").mkdir(exist_ok=True)
    np.save(
        f"results/catboost_{args.task}_rs_test_b30_seed{args.seed}.npy",
        rs_scores)
    np.save(
        f"results/catboost_{args.task}_smac_bohb_test_b30_seed{args.seed}.npy",
        bohb_scores)
    print(f"Saved to results/catboost_{args.task}_*_seed{args.seed}.npy")
