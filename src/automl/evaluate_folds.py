from pathlib import Path
import argparse
import logging

import pandas as pd

from automl.automl import AutoML


logger = logging.getLogger(__name__)


FILE = Path(__file__).absolute().resolve()

DATADIR = FILE.parent / "../../data"


def load_fold(datadir, task, fold):

    fold_dir = datadir / task / str(fold)

    # NOTE: lowercase filenames (x_train / x_test) -- adjust back to
    # X_train / X_test here if your actual files are capitalized.
    X_train = pd.read_parquet(fold_dir / "x_train.parquet")
    y_train = pd.read_parquet(fold_dir / "y_train.parquet").squeeze()
    X_test = pd.read_parquet(fold_dir / "x_test.parquet")
    y_test = pd.read_parquet(fold_dir / "y_test.parquet").squeeze()

    return X_train, y_train, X_test, y_test


def main(task, n_folds, seed, datadir):

    folds = []

    for fold in range(1, n_folds + 1):
        logger.info(f"Loading fold {fold}")
        folds.append(load_fold(datadir, task, fold))

    # Each fold gets its OWN independent HPO search (internal-CV, now via
    # SMAC BO with a Random Forest surrogate + EI acquisition), its own
    # best_params, its own model -- exactly simulating "if this fold were
    # the exam dataset." Nothing is shared or averaged across folds during
    # the search itself.
    scores, params_list, models = AutoML.run_per_fold(
        folds, seed=seed
    )

    logger.info(
        f"{task}: mean R2={scores.mean():.4f} +/- {scores.std():.4f}"
    )

    return scores, params_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datadir", type=Path, default=DATADIR)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Belt-and-suspenders: see run.py for why -- main determinism fix is
    # the LightGBM flags in utils.py, this is a secondary safeguard.
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(
        task=args.task,
        n_folds=args.n_folds,
        seed=args.seed,
        datadir=args.datadir,
    )