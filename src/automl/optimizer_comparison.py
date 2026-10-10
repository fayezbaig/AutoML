"""
optimizer_comparison.py — unified comparison: Random Search, SMAC
black-box, SMAC multi-fidelity (data-fraction), SMAC multi-fidelity
(boosting-rounds), SMAC-BOHB, NePS black-box, NePS multi-fidelity,
Optuna black-box (TPE), and Optuna multi-fidelity (TPE + HyperbandPruner).

EVALUATION PROTOCOL (applies to every mode, for a fair comparison):
  1. Search picks a config using an internal K-fold CV objective on
     X_train (never touches X_test).
  2. The winning config is refit on the FULL X_train (with early-stopping
     round selection for LightGBM).
  3. Final score = R2 on the fold's real, held-out X_test.

NOTE: CatBoost is deliberately NOT included here -- it needs RAW
categorical features (not label-encoded), handled separately via
catboost_bohb.py + preprocessing.py's preprocess_for_algo.

Search space ranges were WIDENED (see ALGORITHMS below) after
investigation into bike_sharing_demand's below-baseline result; XGBoost
was added as a fourth candidate for the same reason.
"""
from automl.preprocessing import encode_categoricals
from automl.data import Dataset
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from pathlib import Path
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

warnings.filterwarnings("ignore")


FILE = Path(__file__).resolve()
DATADIR = FILE.parents[2] / "data"
TASK = "yprop_4_1"
BUDGET = 30
N_SPLITS = 5

LGBM_DETERMINISM_KWARGS = dict(
    deterministic=True,
    force_row_wise=True,
    n_jobs=1)


def build_rf(config, seed=0):
    return RandomForestRegressor(random_state=seed, **config)


def build_lgbm(config, seed=0):
    return lgb.LGBMRegressor(random_state=seed, verbosity=-1, num_threads=1,
                             n_jobs=1, **config)


def build_xgb(config, seed=0):
    import xgboost as xgb
    return xgb.XGBRegressor(random_state=seed, verbosity=0, n_jobs=1, **config)


ALGORITHMS = {
    "random_forest": {
        "builder": build_rf,
        "random_bounds": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 2, 50),
            "min_samples_leaf": ("int", 1, 40),
        },
    },
    "lightgbm": {
        "builder": build_lgbm,
        "random_bounds": {
            "learning_rate": ("log_float", 1e-4, 0.5),
            "num_leaves": ("int", 4, 512),
            "max_depth": ("int", 2, 20),
            "min_child_samples": ("int", 2, 200),
        },
    },
    "xgboost": {
        "builder": build_xgb,
        "random_bounds": {
            "learning_rate": ("log_float", 1e-4, 0.5),
            "max_depth": ("int", 2, 20),
            "min_child_weight": ("int", 1, 40),
            "n_estimators": ("int", 50, 500),
        },
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


def train_and_score(algo_name, config, X_train, y_train, X_val, y_val, seed=0):
    model = ALGORITHMS[algo_name]["builder"](config, seed=seed)
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
        n_splits=N_SPLITS):
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
        scores.append(
            train_and_score(
                algo_name,
                cfg,
                X_tr,
                y_tr,
                X_v,
                y_v,
                seed))
    return float(np.mean(scores))


def refit_final(
        algo_name,
        config,
        X_train,
        y_train,
        seed,
        use_early_stopping=False,
        val_fraction=0.15,
        max_estimators=2000,
        early_stopping_rounds=50):
    if algo_name == "lightgbm" and use_early_stopping:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction, random_state=seed
        )
        probe = lgb.LGBMRegressor(
            random_state=seed,
            n_estimators=max_estimators,
            verbosity=-1,
            num_threads=1,
            n_jobs=1,
            **config)
        probe.fit(
            X_fit, y_fit, eval_set=[
                (X_val, y_val)], callbacks=[
                lgb.early_stopping(
                    early_stopping_rounds, verbose=False)])
        n_rounds = probe.best_iteration_ or max_estimators

        final_model = lgb.LGBMRegressor(
            random_state=seed,
            n_estimators=n_rounds,
            verbosity=-1,
            num_threads=1,
            n_jobs=1,
            **config)
        final_model.fit(X_train, y_train)
        return final_model

    model = ALGORITHMS[algo_name]["builder"](config, seed=seed)
    model.fit(X_train, y_train)
    return model


def run_random_search(algo_name, X_train, y_train, seed, budget):
    rng = np.random.default_rng(seed)
    bounds = ALGORITHMS[algo_name]["random_bounds"]
    best_config, best_score = None, -np.inf
    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        score = internal_cv_score(algo_name, config, X_train, y_train, seed)
        if score > best_score:
            best_score, best_config = score, config
    return best_config


def _build_configspace(algo_name, seed, exclude_n_estimators=False):
    from ConfigSpace import ConfigurationSpace, Integer, Float
    cs = ConfigurationSpace(seed=seed)
    if algo_name == "random_forest":
        params = [
            Integer(
                "max_depth", (2, 50)), Integer(
                "min_samples_leaf", (1, 40))]
        if not exclude_n_estimators:
            params.insert(0, Integer("n_estimators", (50, 500)))
        cs.add(params)
    elif algo_name == "lightgbm":
        cs.add([
            Float("learning_rate", (1e-4, 0.5), log=True),
            Integer("num_leaves", (4, 512)),
            Integer("max_depth", (2, 20)),
            Integer("min_child_samples", (2, 200)),
        ])
    elif algo_name == "xgboost":
        params = [
            Float("learning_rate", (1e-4, 0.5), log=True),
            Integer("max_depth", (2, 20)),
            Integer("min_child_weight", (1, 40)),
        ]
        if not exclude_n_estimators:
            params.append(Integer("n_estimators", (50, 500)))
        cs.add(params)
    return cs


def run_smac_bo(algo_name, X_train, y_train, seed, budget, root_dir):
    from smac import HyperparameterOptimizationFacade, Scenario

    cs = _build_configspace(algo_name, seed)

    def train(config, seed: int = 0):
        return -internal_cv_score(algo_name, dict(config),
                                  X_train, y_train, seed)

    scenario = Scenario(
        cs,
        n_trials=budget,
        seed=seed,
        output_directory=root_dir,
        deterministic=True)
    smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
    incumbent = smac.optimize()
    return dict(incumbent)


def run_smac_mf_datafrac(algo_name, X_train, y_train, seed, budget, root_dir):
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband

    cs = _build_configspace(algo_name, seed)

    def train(config, seed: int = 0, budget: float = 1.0):
        n = max(50, int(len(X_train) * budget))
        X_sub, y_sub = X_train.iloc[:n], y_train.iloc[:n]
        return -internal_cv_score(algo_name, dict(config), X_sub, y_sub, seed)

    scenario = Scenario(
        cs,
        n_trials=budget,
        seed=seed,
        output_directory=root_dir,
        min_budget=0.1,
        max_budget=1.0,
        deterministic=True)
    smac = MultiFidelityFacade(
        scenario, train,
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    incumbent = smac.optimize()
    return dict(incumbent)


def run_smac_mf_rounds(algo_name, X_train, y_train, seed, budget, root_dir,
                       min_budget=60, max_budget=300):
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband

    cs = _build_configspace(algo_name, seed, exclude_n_estimators=True)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score(algo_name,
                                  dict(config),
                                  X_train,
                                  y_train,
                                  seed,
                                  budget=budget)

    scenario = Scenario(
        cs,
        n_trials=budget,
        seed=seed,
        output_directory=root_dir,
        min_budget=min_budget,
        max_budget=max_budget,
        deterministic=True)
    smac = MultiFidelityFacade(
        scenario, train,
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    incumbent = smac.optimize()
    return dict(incumbent)


def run_smac_bohb(algo_name, X_train, y_train, seed, root_dir,
                  min_budget=60, max_budget=300, n_trials=60, eta=3, n_init=10,
                  walltime_limit=None):
    """
    walltime_limit (seconds): hard wall-clock cap on this search. Prevents
    a single algorithm's search from silently consuming the entire 24h
    budget (as CatBoost's search on superconductivity nearly did before
    this fix -- aborted at a projected ~12h with no enforced cap).
    NOTE: per SMAC's own docs, reproducibility is only guaranteed with NO
    time-based stopping criterion -- so walltime_limit trades a small
    amount of determinism for a hard safety guarantee. Leave it None
    for development/comparison runs where reproducibility matters most;
    set it for the actual timed exam-dataset run.
    """
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI

    cs = _build_configspace(algo_name, seed, exclude_n_estimators=True)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score(algo_name,
                                  dict(config),
                                  X_train,
                                  y_train,
                                  seed,
                                  budget=budget)

    scenario_kwargs = dict(
        name="bohb", output_directory=root_dir, n_trials=n_trials,
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
    return dict(incumbent)


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

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=space,
        root_directory=root_dir,
        max_evaluations_total=budget,
        overwrite_working_directory=True,
        optimizer="auto",
    )

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

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=space,
        root_directory=root_dir,
        max_evaluations_total=budget,
        overwrite_working_directory=True,
        optimizer="auto",
    )

    best_config = _extract_best_config(
        root_dir, fidelity_col="config.data_fraction")
    best_config.pop("data_fraction", None)
    return best_config


def run_optuna_bo(algo_name, X_train, y_train, seed, budget, root_dir):
    """Black-box Optuna, default TPE sampler, internal CV objective."""
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
    """Multi-fidelity Optuna -- TPE sampler + HyperbandPruner, fidelity =
    n_estimators (boosting rounds), always trains on full data.
    Only supports lightgbm (matches your other _rounds fidelity modes)."""
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
        min_resource=min_resource, max_resource=max_resource,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner)
    study.optimize(objective, n_trials=budget)

    return study.best_params


MODES = {
    "smac_bo": run_smac_bo,
    "smac_mf_frac": run_smac_mf_datafrac,
    "smac_mf_rounds": run_smac_mf_rounds,
    "neps_bo": run_neps_bo,
    "neps_mf": run_neps_mf,
    "optuna_bo": run_optuna_bo,
    "optuna_mf": run_optuna_mf,
}


def compare(
        algo_name,
        task=TASK,
        folds=None,
        budget=BUDGET,
        seed=0,
        mode="smac_bo",
        use_early_stopping=None):
    folds = folds if folds is not None else list(range(1, 11))
    if use_early_stopping is None:
        use_early_stopping = (algo_name == "lightgbm")

    rs_test_scores, method_test_scores = [], []

    for fold in folds:
        ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
        X_train = encode_categoricals(ds.X_train)
        X_test = encode_categoricals(ds.X_test)
        y_train, y_test = ds.y_train, ds.y_test

        rs_config = run_random_search(
            algo_name, X_train, y_train, seed, budget)
        rs_model = refit_final(
            algo_name,
            rs_config,
            X_train,
            y_train,
            seed,
            use_early_stopping)
        rs_test_scores.append(r2_score(y_test, rs_model.predict(X_test)))

        root_dir = f"results/compare_{algo_name}_{task}_{mode}_fold{fold}"
        if mode == "smac_bohb":
            method_config = run_smac_bohb(
                algo_name, X_train, y_train, seed, root_dir)
        else:
            method_config = MODES[mode](
                algo_name, X_train, y_train, seed, budget, root_dir)

        method_model = refit_final(
            algo_name,
            method_config,
            X_train,
            y_train,
            seed,
            use_early_stopping)
        method_test_scores.append(
            r2_score(
                y_test,
                method_model.predict(X_test)))

    rs_test_scores, method_test_scores = np.array(
        rs_test_scores), np.array(method_test_scores)

    print(f"[{algo_name}/{task}] Random search TEST R2: "
          f"{rs_test_scores.mean():.4f} \u00b1 {rs_test_scores.std():.4f}")
    print(
        f"[{algo_name}/{task}] {mode} TEST R2: "
        f"{method_test_scores.mean():.4f} \u00b1 {method_test_scores.std():.4f}")

    return rs_test_scores, method_test_scores


if __name__ == "__main__":
    from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
    reexec_with_fixed_hash_seed()
    pin_thread_env()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=list(
            ALGORITHMS.keys()),
        required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--mode",
        choices=[
            "smac_bo",
            "smac_mf_frac",
            "smac_mf_rounds",
            "smac_bohb",
            "neps_bo",
            "neps_mf",
            "optuna_bo",
            "optuna_mf"],
        default="smac_bo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rs_scores, method_scores = compare(
        args.algo, task=args.task, mode=args.mode, seed=args.seed)

    Path("results").mkdir(exist_ok=True)
    np.save(
        f"results/{args.algo}_{args.task}_rs_test_b{BUDGET}_seed{args.seed}.npy",
        rs_scores)
    np.save(
        f"results/{args.algo}_{args.task}_{args.mode}_test_b{BUDGET}_seed{args.seed}.npy",
        method_scores)
    print(
        f"Saved TEST-scored per-fold results to results/{args.algo}_{args.task}_*_seed{args.seed}.npy")
