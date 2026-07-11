from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd

from automl.automl import AutoML


logger = logging.getLogger(__name__)


FILE = Path(__file__).absolute().resolve()

DATADIR = FILE.parent / "../../data"


def load_fold(datadir, task, fold):

    fold_dir = datadir / task / str(fold)

    X_train = pd.read_parquet(
        fold_dir / "X_train.parquet"
    )

    y_train = pd.read_parquet(
        fold_dir / "y_train.parquet"
    ).squeeze()

    X_test = pd.read_parquet(
        fold_dir / "X_test.parquet"
    )

    y_test = pd.read_parquet(
        fold_dir / "y_test.parquet"
    ).squeeze()

    return X_train, y_train, X_test, y_test


def main(task, n_folds, seed, datadir):

    folds = []

    for fold in range(1, n_folds + 1):

        logger.info(
            f"Loading fold {fold}"
        )

        folds.append(
            load_fold(
                datadir,
                task,
                fold
            )
        )

    automl = AutoML(
        seed=seed
    )

    # Step 1: multi-fold-averaging HPO (Tier 2) -- picks hyperparameters
    # that hold up well across the several outer folds we were given.
    # This is a practice-dataset-only validation step; the exam dataset
    # only ever gives you one fold, so it uses automl.fit() (Tier 1)
    # instead, not this method.
    automl.fit_multi_fold(
        folds
    )

    # Step 2: Honest per-fold evaluation -- each fold gets its own
    # freshly trained model (same best_params), evaluated only on its
    # own held-out test split.
    scores = automl.evaluate_per_fold(
        folds
    )

    logger.info(
        f"{task}: mean R2={scores.mean():.4f} +/- {scores.std():.4f}"
    )

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=10
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO
    )

    main(
        task=args.task,
        n_folds=args.n_folds,
        seed=args.seed,
        datadir=args.datadir
    )