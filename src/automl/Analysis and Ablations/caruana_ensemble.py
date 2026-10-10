"""
ensemble.py — Caruana-style greedy ensemble selection over LightGBM,
RandomForest, and CatBoost (each tuned via SMAC-BOHB), without TabPFN
pending confirmation on submission eligibility. Now times prediction
cost per model for Pareto-front use.
"""
import time
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from automl.data import Dataset
from automl.optimizer_comparison import DATADIR, encode_categoricals, run_smac_bohb, refit_final
from automl.extended_portfolio import preprocess_for_algo
from automl.catboost_bohb import run_catboost_bohb


def _detect_device():
    """Auto-detect best available device: CUDA > Apple MPS > CPU.
    Avoids hardcoding a platform-specific device that would silently
    break (or need manual editing) on teammates' machines, the grading
    machine, or a cluster (e.g. bwUniCluster/NEMO2)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _timed_predict(model, X):
    t0 = time.time()
    preds = model.predict(X)
    elapsed = time.time() - t0
    return preds, elapsed


def get_candidate_predictions_from_arrays(X, y, seed=0):
    """Same as get_candidate_predictions, but takes raw X, y directly
    instead of loading via Dataset.load -- lets any dataset (not just
    our 5 practice datasets) flow through the SAME production ensemble
    pipeline used everywhere else, rather than duplicating logic in a
    separate script."""
    candidates = {}

    X_full = encode_categoricals(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y, test_size=0.2, random_state=seed
    )

    for algo_name in ("lightgbm", "random_forest"):
        best_config = run_smac_bohb(
            algo_name, X_train, y_train, seed,
            root_dir=f"results/scaling_test_{algo_name}",
        )
        model = refit_final(algo_name, best_config, X_train, y_train, seed,
                            use_early_stopping=(algo_name == "lightgbm"))
        val_preds = model.predict(X_val)
        score = r2_score(y_val, val_preds)
        print(f"[{algo_name}] val R2={score:.4f}")
        candidates[algo_name] = {"model": model, "val_preds": val_preds}

    from catboost import CatBoostRegressor
    X_train_cb, X_val_cb, cat_features = preprocess_for_algo("catboost", X, X)
    X_train_cb_split, X_val_cb, y_train_cb, y_val_cb = train_test_split(
        X_train_cb, y, test_size=0.2, random_state=seed
    )
    best_config_cb = run_catboost_bohb(
        X_train_cb_split, y_train_cb, cat_features, seed,
        root_dir="results/scaling_test_catboost",
    )
    cb_model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                 cat_features=cat_features, **best_config_cb)
    cb_model.fit(X_train_cb_split, y_train_cb)
    val_preds_cb = cb_model.predict(X_val_cb)
    score_cb = r2_score(y_val_cb, val_preds_cb)
    print(f"[catboost] val R2={score_cb:.4f}")
    candidates["catboost"] = {"model": cb_model, "val_preds": val_preds_cb}

    from tabpfn import TabPFNRegressor
    from automl.pareto_front import estimate_and_check_tabpfn_feasible
    tabpfn_model = TabPFNRegressor()
    tabpfn_model.fit(X_train, y_train)
    feasible, est_cost = estimate_and_check_tabpfn_feasible(
        tabpfn_model, X_val, n_test_rows=len(X_val), max_predict_seconds=300
    )
    if feasible:
        val_preds_tp = tabpfn_model.predict(X_val)
        score_tp = r2_score(y_val, val_preds_tp)
        print(f"[tabpfn] val R2={score_tp:.4f} (estimated cost was {est_cost:.1f}s)")
        candidates["tabpfn"] = {"model": tabpfn_model, "val_preds": val_preds_tp}
    else:
        print(f"[tabpfn] EXCLUDED -- estimated cost {est_cost:.1f}s exceeds 300s cutoff")

    return candidates, y_val


def get_candidate_predictions(task, fold=1, seed=0):
    ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
    candidates = {}

    X_full = encode_categoricals(ds.X_train)
    X_test_shared = encode_categoricals(ds.X_test)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, ds.y_train, test_size=0.2, random_state=seed
    )

    for algo_name in ("lightgbm", "random_forest"):
        best_config = run_smac_bohb(
            algo_name, X_train, y_train, seed,
            root_dir=f"results/ensemble_{algo_name}_{task}",
        )
        model = refit_final(algo_name, best_config, X_train, y_train, seed,
                            use_early_stopping=(algo_name == "lightgbm"))
        val_preds, _ = _timed_predict(model, X_val)
        test_preds, predict_time = _timed_predict(model, X_test_shared)
        score = r2_score(y_val, val_preds)
        print(f"[{algo_name}] val R2={score:.4f}  predict_time={predict_time:.4f}s (on {len(X_test_shared)} test rows)")
        candidates[algo_name] = {"model": model, "val_preds": val_preds, "test_preds": test_preds,
                                 "predict_time": predict_time, "n_test_rows": len(X_test_shared)}

    from catboost import CatBoostRegressor
    X_train_cb, X_test_cb, cat_features = preprocess_for_algo("catboost", ds.X_train, ds.X_test)
    X_train_cb_split, X_val_cb, y_train_cb, y_val_cb = train_test_split(
        X_train_cb, ds.y_train, test_size=0.2, random_state=seed
    )
    best_config_cb = run_catboost_bohb(
        X_train_cb_split, y_train_cb, cat_features, seed,
        root_dir=f"results/ensemble_catboost_{task}",
    )
    cb_model = CatBoostRegressor(random_state=seed, verbose=False, thread_count=1,
                                 cat_features=cat_features, **best_config_cb)
    cb_model.fit(X_train_cb_split, y_train_cb)
    val_preds_cb, _ = _timed_predict(cb_model, X_val_cb)
    test_preds_cb, predict_time_cb = _timed_predict(cb_model, X_test_cb)
    score_cb = r2_score(y_val_cb, val_preds_cb)
    print(f"[catboost] val R2={score_cb:.4f}  predict_time={predict_time_cb:.4f}s (on {len(X_test_cb)} test rows)")
    candidates["catboost"] = {"model": cb_model, "val_preds": val_preds_cb, "test_preds": test_preds_cb,
                              "predict_time": predict_time_cb, "n_test_rows": len(X_test_cb)}

    assert np.allclose(y_val.values, y_val_cb.values), \
        "Validation split mismatch between shared and catboost preprocessing!"

    # --- TabPFN: zero-tuning pretrained transformer, single fit/score ---
    # Per TabPFN's own README: do NOT scale or one-hot encode features --
    # feed it the RAW train/val/test splits directly (categorical dtype
    # columns are handled natively, same principle as CatBoost).
    try:
        from tabpfn import TabPFNRegressor
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            ds.X_train, ds.y_train, test_size=0.2, random_state=seed
        )
        X_test_raw = ds.X_test
        tabpfn_model = TabPFNRegressor(device=_detect_device())
        tabpfn_model.fit(X_train_raw, y_train_raw)

        val_preds_tp, _ = _timed_predict(tabpfn_model, X_val_raw)
        test_preds_tp, predict_time_tp = _timed_predict(tabpfn_model, X_test_raw)
        score_tp = r2_score(y_val, val_preds_tp)
        print(f"[tabpfn] val R2={score_tp:.4f}  predict_time={predict_time_tp:.4f}s "
              f"(on {len(X_test_shared)} test rows)")
        candidates["tabpfn"] = {"model": tabpfn_model, "val_preds": val_preds_tp, "test_preds": test_preds_tp,
                                "predict_time": predict_time_tp, "n_test_rows": len(X_test_shared)}
    except Exception as e:
        print(f"TabPFN unavailable ({e}), skipping.")

    return candidates, y_val, ds.y_test


def caruana_ensemble_select(candidates, y_val, n_rounds=10):
    names = list(candidates.keys())
    ensemble = []
    current_sum = np.zeros_like(np.asarray(y_val), dtype=float)

    for round_i in range(n_rounds):
        best_name, best_score = None, -np.inf
        for name in names:
            trial_sum = current_sum + candidates[name]["val_preds"]
            trial_avg = trial_sum / (len(ensemble) + 1)
            score = r2_score(y_val, trial_avg)
            if score > best_score:
                best_score, best_name = score, name

        ensemble.append(best_name)
        current_sum += candidates[best_name]["val_preds"]
        print(f"Round {round_i + 1}: added {best_name:15s} -> ensemble val R2={best_score:.4f}")

    return ensemble


def ensemble_predict(ensemble_names, candidates, key="test_preds"):
    stacked = np.stack([candidates[name][key] for name in ensemble_names])
    return stacked.mean(axis=0)


def run_ensemble_comparison(task, fold=1, seed=0, n_rounds=10):
    print(f"=== {task}: tuning portfolio (LightGBM, RandomForest, CatBoost) ===")
    candidates, y_val, y_test = get_candidate_predictions(task, fold, seed)

    print(f"\n=== {task}: greedy ensemble selection (on validation) ===")
    ensemble_names = caruana_ensemble_select(candidates, y_val, n_rounds)

    best_single_name = max(candidates, key=lambda n: r2_score(y_val, candidates[n]["val_preds"]))
    best_single_test_r2 = r2_score(y_test, candidates[best_single_name]["test_preds"])

    ensemble_test_preds = ensemble_predict(ensemble_names, candidates)
    ensemble_test_r2 = r2_score(y_test, ensemble_test_preds)
    ensemble_predict_time = sum(candidates[n]["predict_time"] for n in set(ensemble_names))

    for name, c in candidates.items():
        c["val_r2"] = r2_score(y_val, c["val_preds"])

    print(f"\n=== {task}: FINAL TEST comparison ===")
    print(f"Best single model ({best_single_name}): TEST R2 = {best_single_test_r2:.4f}, "
          f"predict_time={candidates[best_single_name]['predict_time']:.4f}s")
    print(f"Caruana ensemble {ensemble_names}: TEST R2 = {ensemble_test_r2:.4f}, "
          f"predict_time(sum of unique models)={ensemble_predict_time:.4f}s")

    print(f"\n=== Per-model prediction cost (for Pareto front) ===")
    for name, c in candidates.items():
        print(f"  {name:15s} R2(val)={r2_score(y_val, c['val_preds']):.4f}  "
              f"predict_time={c['predict_time']:.4f}s  n_rows={c['n_test_rows']}")

    return {"best_single_name": best_single_name, "best_single": best_single_test_r2,
            "ensemble": ensemble_test_r2, "ensemble_composition": ensemble_names,
            "candidates": candidates}


if __name__ == "__main__":
    from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
    reexec_with_fixed_hash_seed()
    pin_thread_env()
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_ensemble_comparison(args.task, fold=args.fold, seed=args.seed)

    summary = {
        "task": args.task,
        "best_single_name": result["best_single_name"],
        "best_single_r2": result["best_single"],
        "ensemble_r2": result["ensemble"],
        "ensemble_composition": result["ensemble_composition"],
        "per_model": {
            name: {
                "val_r2": c["val_r2"],
                "predict_time": c["predict_time"],
                "n_test_rows": c["n_test_rows"],
            }
            for name, c in result["candidates"].items()
        },
    }
    ensemble_predict_time = sum(
        result["candidates"][n]["predict_time"] for n in set(result["ensemble_composition"])
    )
    summary["ensemble_predict_time"] = ensemble_predict_time

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"ensemble_summary_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")
