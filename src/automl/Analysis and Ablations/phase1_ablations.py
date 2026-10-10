"""
phase1_ablations.py — search methods and algorithms tested during
Phase I development but NOT used in the final production pipeline.
Retained for reproducibility of reported comparison results. NOT
imported or called by automl.core's fit()/predict() -- adds zero
computational cost to the actual exam-dataset evaluation.

Contents and conclusions:
  - NePS (black-box + multi-fidelity): comparable to SMAC-BOHB once
    evaluated under our corrected protocol (internal CV + true test
    scoring).
  - Optuna (black-box + multi-fidelity): competitive with SMAC-BOHB,
    even won on wine_quality -- not adopted mainly because SMAC-BOHB
    was already working and no significant advantage was shown.
  - XGBoost: competitive with tuned LightGBM even with only random
    search, but never clearly better.
  - ElasticNet: substantially underperformed every tree-based method
    on every dataset tested.
  - Meta-learning warm-starting: leave-one-out validated; HURT
    performance on 4/5 datasets -- an honest negative result.
"""
from automl.search_spaces import RANDOM_BOUNDS, sample_random_config
from automl.preprocessing import encode_categoricals
from automl.data import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from pathlib import Path
import numpy as np
import json
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")


FILE = Path(__file__).resolve()
DATADIR = FILE.parents[2] / "data"


def train_and_score(algo_name, config, X_train, y_train, X_val, y_val):
    if algo_name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            random_state=0,
            verbosity=-1,
            num_threads=1,
            n_jobs=1,
            **config)
    elif algo_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=0, **config)
    else:
        raise ValueError(algo_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return r2_score(y_val, preds)


def internal_cv_score(
        algo_name,
        config,
        X_train,
        y_train,
        seed,
        budget=None,
        n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)
    cfg = dict(config)
    if budget is not None:
        cfg["n_estimators"] = int(budget)
    scores = []
    for tr_idx, val_idx in kf.split(X_reset):
        X_tr, X_v = X_reset.iloc[tr_idx], X_reset.iloc[val_idx]
        y_tr, y_v = y_reset.iloc[tr_idx], y_reset.iloc[val_idx]
        scores.append(train_and_score(algo_name, cfg, X_tr, y_tr, X_v, y_v))
    return float(np.mean(scores))


INT_PARAMS = {
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "n_estimators",
    "min_samples_leaf"}


def _extract_best_config(root_dir, fidelity_col=None):
    import neps
    full_df, _ = neps.status(root_dir, print_summary=False)
    df = full_df
    if fidelity_col is not None and fidelity_col in df.columns:
        max_fid = df[fidelity_col].max()
        high_fid = df[df[fidelity_col] >= max_fid * 0.99]
        if len(high_fid) > 0:
            df = high_fid
    best_row = df.loc[df["objective_to_minimize"].idxmin()]
    config_cols = [c for c in full_df.columns if c.startswith("config.")]
    best_config = {}
    for c in config_cols:
        name = c.replace("config.", "")
        val = best_row[c]
        best_config[name] = int(val) if name in INT_PARAMS else val
    return best_config


def run_neps_bo(algo_name, X_train, y_train, seed, budget, root_dir):
    import neps
    import random
    random.seed(seed)
    np.random.seed(seed)

    def evaluate_pipeline(**config):
        score = internal_cv_score(algo_name, config, X_train, y_train, seed)
        return {"objective_to_minimize": -score}

    if algo_name == "lightgbm":
        space = dict(
            learning_rate=neps.Float(lower=1e-3, upper=0.3, log=True),
            num_leaves=neps.Integer(lower=8, upper=256),
            max_depth=neps.Integer(lower=3, upper=12),
            min_child_samples=neps.Integer(lower=5, upper=100),
        )
    else:
        space = dict(
            n_estimators=neps.Integer(lower=50, upper=300),
            max_depth=neps.Integer(lower=2, upper=32),
            min_samples_leaf=neps.Integer(lower=1, upper=20),
        )

    neps.run(evaluate_pipeline=evaluate_pipeline, pipeline_space=space,
             root_directory=root_dir, max_evaluations_total=budget,
             overwrite_working_directory=True, optimizer="auto")
    return _extract_best_config(root_dir)


def run_neps_mf(algo_name, X_train, y_train, seed, budget, root_dir,
                min_fraction=0.1, max_fraction=1.0):
    import neps
    import random
    random.seed(seed)
    np.random.seed(seed)

    def evaluate_pipeline(**config):
        cfg = dict(config)
        frac = cfg.pop("data_fraction")
        n = max(50, int(len(X_train) * frac))
        X_sub, y_sub = X_train.iloc[:n], y_train.iloc[:n]
        score = internal_cv_score(algo_name, cfg, X_sub, y_sub, seed)
        return {"objective_to_minimize": -score}

    if algo_name == "lightgbm":
        space = dict(
            learning_rate=neps.Float(
                lower=1e-3,
                upper=0.3,
                log=True),
            num_leaves=neps.Integer(
                lower=8,
                upper=256),
            max_depth=neps.Integer(
                lower=3,
                upper=12),
            min_child_samples=neps.Integer(
                lower=5,
                upper=100),
            data_fraction=neps.Float(
                lower=min_fraction,
                upper=max_fraction,
                is_fidelity=True),
        )
    else:
        space = dict(
            n_estimators=neps.Integer(
                lower=50, upper=300), max_depth=neps.Integer(
                lower=2, upper=32), min_samples_leaf=neps.Integer(
                lower=1, upper=20), data_fraction=neps.Float(
                    lower=min_fraction, upper=max_fraction, is_fidelity=True), )

    neps.run(evaluate_pipeline=evaluate_pipeline, pipeline_space=space,
             root_directory=root_dir, max_evaluations_total=budget,
             overwrite_working_directory=True, optimizer="auto")
    best_config = _extract_best_config(
        root_dir, fidelity_col="config.data_fraction")
    best_config.pop("data_fraction", None)
    return best_config


def run_optuna_bo(algo_name, X_train, y_train, seed, budget, root_dir):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if algo_name == "lightgbm":
            config = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    1e-3,
                    0.3,
                    log=True),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    8,
                    256),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    12),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    5,
                    100),
            }
        else:
            config = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 50, 300), "max_depth": trial.suggest_int(
                    "max_depth", 2, 32), "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 20), }
        return internal_cv_score(algo_name, config, X_train, y_train, seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=budget)
    return study.best_params


def run_optuna_mf(algo_name, X_train, y_train, seed, budget, root_dir,
                  min_resource=60, max_resource=300, reduction_factor=3):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        config = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-3,
                0.3,
                log=True),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                8,
                256),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                5,
                100),
        }
        resource = min_resource
        score = None
        step = 0
        while True:
            score = internal_cv_score(
                algo_name,
                config,
                X_train,
                y_train,
                seed,
                budget=resource)
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if resource >= max_resource:
                break
            resource = min(max_resource, resource * reduction_factor)
            step += 1
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner)
    study.optimize(objective, n_trials=budget)
    return study.best_params


def build_xgboost(config, seed=0):
    import xgboost as xgb
    return xgb.XGBRegressor(random_state=seed, verbosity=0, n_jobs=1, **config)


def build_elasticnet(config, seed=0):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(random_state=seed, **config)


def run_random_search_extended(algo_name, X_train, y_train, seed, budget):
    rng = np.random.default_rng(seed)
    bounds = RANDOM_BOUNDS[algo_name]
    best_config, best_score = None, -np.inf
    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        X_reset = X_train.reset_index(
            drop=True) if hasattr(
            X_train, "reset_index") else X_train
        y_reset = y_train.reset_index(drop=True)
        scores = []
        for tr_idx, val_idx in kf.split(X_reset):
            X_tr = X_reset.iloc[tr_idx] if hasattr(
                X_reset, "iloc") else X_reset[tr_idx]
            X_v = X_reset.iloc[val_idx] if hasattr(
                X_reset, "iloc") else X_reset[val_idx]
            y_tr, y_v = y_reset.iloc[tr_idx], y_reset.iloc[val_idx]
            m = build_xgboost(
                config,
                seed=seed) if algo_name == "xgboost" else build_elasticnet(
                config,
                seed=seed)
            m.fit(X_tr, y_tr)
            scores.append(r2_score(y_v, m.predict(X_v)))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_config = score, config
    return best_config


def compare_extended_portfolio(algo_name, task, folds=None, budget=30, seed=0):
    from automl.preprocessing import preprocess_for_algo
    folds = folds if folds is not None else list(range(1, 11))
    test_scores = []
    for fold in folds:
        ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
        if algo_name == "elasticnet":
            X_train, X_test, _ = preprocess_for_algo(
                "elasticnet", ds.X_train, ds.X_test)
        else:
            X_train = encode_categoricals(ds.X_train)
            X_test = encode_categoricals(ds.X_test)
        y_train, y_test = ds.y_train, ds.y_test
        best_config = run_random_search_extended(
            algo_name, X_train, y_train, seed, budget)
        model = build_xgboost(
            best_config,
            seed=seed) if algo_name == "xgboost" else build_elasticnet(
            best_config,
            seed=seed)
        model.fit(X_train, y_train)
        test_scores.append(r2_score(y_test, model.predict(X_test)))
    test_scores = np.array(test_scores)
    print(f"[{algo_name}/{task}] Random search TEST R2: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    return test_scores


# ============================================================
# Meta-learning warm-starting (tested, honest negative result --
# HURT performance on 4/5 datasets in leave-one-out validation)
# ============================================================

META_DB_PATH = Path(__file__).parent / "meta_database.json"


def compute_meta_features(X, y):
    n_rows, n_cols = X.shape
    n_categorical = sum(1 for c in X.columns if X[c].nunique() < 20)
    return {
        "n_rows": int(n_rows), "n_features": int(n_cols),
        "frac_categorical": float(n_categorical / n_cols),
        "target_std": float(y.std()), "target_skew": float(y.skew()),
    }


def _normalize(vec, mins, maxs):
    return [(v - mn) / (mx - mn) if mx > mn else 0.0 for v,
            mn, mx in zip(vec, mins, maxs)]


def build_meta_database(tasks, algo_name="lightgbm", fold=1, seed=0):
    """Run search once per practice dataset, record meta-features and
    winning config. One-time Phase I cost."""
    from automl.search_methods import run_smac_bohb

    db = {}
    for task in tasks:
        print(f"Building meta-entry for {task}...")
        ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
        X_train = encode_categoricals(ds.X_train)
        meta_features = compute_meta_features(X_train, ds.y_train)
        best_config = run_smac_bohb(algo_name, X_train, ds.y_train, seed,
                                    root_dir=f"results/metadb_{task}")
        db[task] = {"meta_features": meta_features, "best_config": best_config}

    with open(META_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"Saved meta-database to {META_DB_PATH}")
    return db


def warm_start_config(new_X, new_y, db_path=META_DB_PATH):
    with open(db_path) as f:
        db = json.load(f)

    keys = [
        "n_rows",
        "n_features",
        "frac_categorical",
        "target_std",
        "target_skew"]
    new_meta = compute_meta_features(new_X, new_y)
    new_vec = [new_meta[k] for k in keys]

    all_vecs = [[entry["meta_features"][k]
                 for k in keys] for entry in db.values()]
    mins = [min(v[i] for v in all_vecs) for i in range(len(keys))]
    maxs = [max(v[i] for v in all_vecs) for i in range(len(keys))]
    new_norm = np.array(_normalize(new_vec, mins, maxs))

    best_task, best_dist = None, float("inf")
    for task, entry in db.items():
        vec = [entry["meta_features"][k] for k in keys]
        norm = np.array(_normalize(vec, mins, maxs))
        dist = np.linalg.norm(new_norm - norm)
        if dist < best_dist:
            best_dist, best_task = dist, task

    print(f"Nearest neighbor: {best_task} (distance={best_dist:.3f})")
    return db[best_task]["best_config"], best_task


def leave_one_out_validation(
        all_tasks,
        algo_name="lightgbm",
        fold=1,
        seed=0,
        budget=30):
    """RESULT: warm-starting HURT performance on 4/5 datasets (mean
    delta -0.0034). NOT used in the final pipeline."""
    from automl.search_methods import run_smac_bo, refit_final
    from automl.search_spaces import build_configspace

    results = {}
    with open(META_DB_PATH) as f:
        full_db = json.load(f)

    for held_out in all_tasks:
        remaining = [t for t in all_tasks if t != held_out]
        print(
            f"\n=== Held out: {held_out} (meta-DB built from {remaining}) ===")
        db = {task: full_db[task] for task in remaining}

        ds_test = Dataset.load(datadir=DATADIR, task=held_out, fold=fold)
        X_test_full = encode_categoricals(ds_test.X_train)
        y_test_full = ds_test.y_train
        X_holdout_test = encode_categoricals(ds_test.X_test)
        y_holdout_test = ds_test.y_test

        keys = [
            "n_rows",
            "n_features",
            "frac_categorical",
            "target_std",
            "target_skew"]
        new_meta = compute_meta_features(X_test_full, y_test_full)
        new_vec = [new_meta[k] for k in keys]
        all_vecs = [[db[t]["meta_features"][k]
                     for k in keys] for t in remaining]
        mins = [min(v[i] for v in all_vecs) for i in range(len(keys))]
        maxs = [max(v[i] for v in all_vecs) for i in range(len(keys))]
        new_norm = np.array(_normalize(new_vec, mins, maxs))

        best_task, best_dist = None, float("inf")
        for task in remaining:
            vec = [db[task]["meta_features"][k] for k in keys]
            norm = np.array(_normalize(vec, mins, maxs))
            dist = np.linalg.norm(new_norm - norm)
            if dist < best_dist:
                best_dist, best_task = dist, task
        warm_config = db[best_task]["best_config"]
        print(f"Nearest neighbor: {best_task} (distance={best_dist:.3f})")

        from smac import HyperparameterOptimizationFacade, Scenario
        from smac.initial_design import DefaultInitialDesign

        def train(config, seed: int = 0):
            return -internal_cv_score(algo_name,
                                      dict(config),
                                      X_test_full,
                                      y_test_full,
                                      seed)

        cs_warm = build_configspace(algo_name, seed)
        for hp_name, value in warm_config.items():
            if hp_name in cs_warm:
                cs_warm[hp_name].default_value = value
        scenario_warm = Scenario(
            cs_warm,
            n_trials=budget,
            seed=seed,
            output_directory=f"results/loo_warm_{held_out}",
            deterministic=True)
        initial_design = DefaultInitialDesign(scenario_warm, n_configs=1)
        smac_warm = HyperparameterOptimizationFacade(
            scenario_warm, train, initial_design=initial_design, overwrite=True)
        warm_incumbent = smac_warm.optimize()

        warm_model = refit_final(
            algo_name,
            dict(warm_incumbent),
            X_test_full,
            y_test_full,
            seed,
            use_early_stopping=True)
        warm_score = r2_score(
            y_holdout_test,
            warm_model.predict(X_holdout_test))

        cold_config = run_smac_bo(
            algo_name,
            X_test_full,
            y_test_full,
            seed,
            budget,
            root_dir=f"results/loo_cold_{held_out}")
        cold_model = refit_final(
            algo_name,
            cold_config,
            X_test_full,
            y_test_full,
            seed,
            use_early_stopping=True)
        cold_score = r2_score(
            y_holdout_test,
            cold_model.predict(X_holdout_test))

        print(
            f"[{held_out}] Cold-start R2: {cold_score:.4f}  |  Warm-start R2: {warm_score:.4f}")
        results[held_out] = {
            "cold": cold_score,
            "warm": warm_score,
            "nearest": best_task}

    return results


if __name__ == "__main__":
    print("phase1_ablations.py -- import the functions you need directly.")
