"""baseline.py — simple 80/20 holdout algorithm-selection baseline.

For each outer fold:

1. Split the outer-training data once into 80% search-training and
   20% validation data.
2. Random-search every algorithm using that same holdout split.
3. Select the algorithm whose best configuration has the highest
   validation R^2.
4. Refit the selected configuration on the complete outer-training set.
5. Evaluate on the untouched outer-test set.
6. Repeat over all requested outer folds and average test R^2.

There is no internal K-fold cross-validation in this baseline.

The script reuses the project's existing:
    - preprocessing functions;
    - algorithm builders;
    - random-search spaces and sampler;
    - final refit function.

Usage:
    python baseline.py --task bike_sharing_demand
    python baseline.py --task wine_quality --budget 30 --seed 0
    python baseline.py --task wine_quality --folds 1,2,3
    python baseline.py --task exam_dataset
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
reexec_with_fixed_hash_seed()
pin_thread_env()

import argparse
import logging
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from automl.data import Dataset
from automl.optimizer_comparison import ALGORITHMS, refit_final
from automl.preprocessing import (
    encode_categoricals,
    handle_missing_values,
    preprocess_for_algo,
)
from automl.search_spaces import RANDOM_BOUNDS, sample_random_config


logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"

PORTFOLIO = ("lightgbm", "random_forest", "catboost", "xgboost")
DEFAULT_FOLDS = list(range(1, 11))
VALIDATION_FRACTION = 0.20


def preprocess_tree(X_train, X_other):
    """Use the project's standard tree-model preprocessing."""
    X_train, X_other = handle_missing_values(X_train, X_other)
    return encode_categoricals(X_train), encode_categoricals(X_other)


def random_search_tree(
    algo_name,
    X_train,
    y_train,
    X_val,
    y_val,
    seed,
    budget,
):
    """Random search using one fixed validation split."""
    rng = np.random.default_rng(seed)
    bounds = ALGORITHMS[algo_name]["random_bounds"]

    best_config = None
    best_score = -np.inf

    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        model = ALGORITHMS[algo_name]["builder"](config, seed=seed)
        model.fit(X_train, y_train)
        score = r2_score(y_val, model.predict(X_val))

        if score > best_score:
            best_config = config
            best_score = score

    if best_config is None:
        raise RuntimeError(f"No configuration found for {algo_name}.")

    return best_config, float(best_score)


def random_search_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_features,
    seed,
    budget,
):
    """CatBoost random search using the same fixed validation split."""
    rng = np.random.default_rng(seed)
    bounds = RANDOM_BOUNDS["catboost"]

    best_config = None
    best_score = -np.inf

    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        model = CatBoostRegressor(
            random_state=seed,
            verbose=False,
            thread_count=1,
            cat_features=cat_features,
            **config,
        )
        model.fit(X_train, y_train)
        score = r2_score(y_val, model.predict(X_val))

        if score > best_score:
            best_config = config
            best_score = score

    if best_config is None:
        raise RuntimeError("No configuration found for catboost.")

    return best_config, float(best_score)


def tune_all_algorithms(X_train, y_train, seed, budget):
    """Tune all algorithms using one shared 80/20 holdout split."""
    X_search, X_val, y_search, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        shuffle=True,
    )

    X_search_tree, X_val_tree = preprocess_tree(X_search, X_val)
    X_search_cb, X_val_cb, cat_features = preprocess_for_algo(
        "catboost",
        X_search,
        X_val,
    )

    results = {}

    for algo_name in PORTFOLIO:
        if algo_name == "catboost":
            config, score = random_search_catboost(
                X_search_cb,
                y_search,
                X_val_cb,
                y_val,
                cat_features,
                seed,
                budget,
            )
        else:
            config, score = random_search_tree(
                algo_name,
                X_search_tree,
                y_search,
                X_val_tree,
                y_val,
                seed,
                budget,
            )

        results[algo_name] = {
            "config": config,
            "validation_r2": score,
        }

        logger.info(
            "    [%s] best 80/20 validation R^2 = %.4f",
            algo_name,
            score,
        )

    return results


def refit_and_predict(
    algo_name,
    config,
    X_train,
    y_train,
    X_test,
    seed,
):
    """Refit the selected algorithm on all outer-training data."""
    if algo_name == "catboost":
        X_train_cb, X_test_cb, cat_features = preprocess_for_algo(
            "catboost",
            X_train,
            X_test,
        )

        model = CatBoostRegressor(
            random_state=seed,
            verbose=False,
            thread_count=1,
            cat_features=cat_features,
            **config,
        )
        model.fit(X_train_cb, y_train)
        return np.asarray(model.predict(X_test_cb))

    X_train_tree, X_test_tree = preprocess_tree(X_train, X_test)

    model = refit_final(
        algo_name,
        config,
        X_train_tree,
        y_train,
        seed,
        use_early_stopping=(algo_name == "lightgbm"),
    )
    return np.asarray(model.predict(X_test_tree))


def run_fold(task, fold, seed, budget, datadir, results_dir):
    """Run tuning, selection, refit, and evaluation for one outer fold."""
    logger.info("[%s] outer fold %d", task, fold)

    dataset = Dataset.load(
        datadir=datadir,
        task=task,
        fold=fold,
    )

    tuned = tune_all_algorithms(
        dataset.X_train,
        dataset.y_train,
        seed,
        budget,
    )

    selected_algo = max(
        tuned,
        key=lambda name: tuned[name]["validation_r2"],
    )
    selected_config = tuned[selected_algo]["config"]
    selected_validation_r2 = tuned[selected_algo]["validation_r2"]

    logger.info(
        "    selected algorithm = %s "
        "(80/20 validation R^2 = %.4f)",
        selected_algo,
        selected_validation_r2,
    )

    predictions = refit_and_predict(
        selected_algo,
        selected_config,
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        seed,
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = (
        results_dir
        / f"baseline_holdout_{task}_fold{fold}_predictions_seed{seed}.npy"
    )
    np.save(prediction_path, predictions)

    test_r2 = None
    if dataset.y_test is not None:
        test_r2 = float(r2_score(dataset.y_test, predictions))
        logger.info(
            "    selected %s outer-test R^2 = %.4f",
            selected_algo,
            test_r2,
        )

    return {
        "fold": fold,
        "selected_algorithm": selected_algo,
        "validation_r2": selected_validation_r2,
        "test_r2": test_r2,
        "prediction_path": prediction_path,
        "all_validation_scores": {
            name: result["validation_r2"]
            for name, result in tuned.items()
        },
    }


def main(task, seed, budget, folds, datadir, results_dir):
    if budget < 1:
        raise ValueError("--budget must be at least 1.")

    if task == "exam_dataset":
        folds = [1]

    fold_results = []

    for fold in folds:
        result = run_fold(
            task,
            fold,
            seed,
            budget,
            datadir,
            results_dir,
        )
        fold_results.append(result)

        validation_summary = ", ".join(
            f"{name}={score:.4f}"
            for name, score in sorted(
                result["all_validation_scores"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        print(
            f"[{task}] outer fold {fold}: "
            f"selected={result['selected_algorithm']}, "
            f"validation R^2={result['validation_r2']:.4f}"
        )
        print(f"  validation scores: {validation_summary}")

        if result["test_r2"] is not None:
            print(f"  outer-test R^2={result['test_r2']:.4f}")
        else:
            print(f"  predictions saved to {result['prediction_path']}")

    test_scores = np.asarray(
        [
            result["test_r2"]
            for result in fold_results
            if result["test_r2"] is not None
        ],
        dtype=float,
    )

    selected_algorithms = np.asarray(
        [result["selected_algorithm"] for result in fold_results],
        dtype=object,
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        results_dir
        / f"baseline_holdout_{task}_selected_algorithms_seed{seed}.npy",
        selected_algorithms,
    )

    if len(test_scores) > 0:
        scores_path = (
            results_dir
            / f"baseline_holdout_{task}_test_r2_b{budget}_seed{seed}.npy"
        )
        np.save(scores_path, test_scores)

        print(
            f"\n[{task}] mean outer-test R^2 over "
            f"{len(test_scores)} fold(s): "
            f"{test_scores.mean():.4f} +/- {test_scores.std():.4f}"
        )
        print(f"Saved per-fold test scores to {scores_path}")
    else:
        print(
            f"\n[{task}] no test labels available; "
            "mean outer-test R^2 was not computed."
        )

    print("\nSelected algorithm per outer fold:")
    for result in fold_results:
        print(
            f"  fold {result['fold']:>2}: "
            f"{result['selected_algorithm']:15s} "
            f"(validation R^2={result['validation_r2']:.4f})"
        )

    return fold_results, test_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "bike_sharing_demand",
            "brazilian_houses",
            "superconductivity",
            "wine_quality",
            "yprop_4_1",
            "exam_dataset",
        ],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=30,
        help="Random-search trials per algorithm.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated outer folds, for example 1,2,3.",
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if args.folds:
        folds = [
            int(value.strip())
            for value in args.folds.split(",")
            if value.strip()
        ]
    elif args.task == "exam_dataset":
        folds = [1]
    else:
        folds = DEFAULT_FOLDS

    main(
        task=args.task,
        seed=args.seed,
        budget=args.budget,
        folds=folds,
        datadir=args.datadir,
        results_dir=args.results_dir,
    )