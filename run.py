"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

For the practice datasets you are given access to y_test; however,
for the exam dataset, you will not have access to these labels.
You will need to output your predictions for X_test to a file,
which we will grade using GitHub Classroom.
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from automl.data import Dataset
from automl.automl import AutoML
import argparse



import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
):
    """
    Single-fold path. This is the ONE that matches the exam dataset:
    only ever one (X_train, y_train, X_test) is available, so HPO here
    uses internal-CV (Tier 1, automl.fit) -- it never assumes outer
    folds exist.
    """
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Fitting AutoML (Tier 1: internal-CV HPO)")

    automl = AutoML(seed=seed)
    automl.fit(dataset.X_train, dataset.y_train)
    test_preds: np.ndarray = automl.predict(dataset.X_test)

    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    logger.info("Writing predictions to disk")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info(f"No test labels (y_test) for task '{task}'")


def evaluate_across_folds(
    task: str,
    n_folds: int,
    seed: int,
    datadir: Path,
    output_path: Path,
):
    """
    Multi-fold path. Only valid for the practice datasets (which give
    you several outer folds). HPO here uses multi-fold-averaging
    (Tier 2, automl.fit_multi_fold) to select hyperparameters that hold
    up across different train/test splits, THEN evaluate_per_fold trains
    one fresh model per fold with those hyperparameters and reports the
    honest per-fold test R2 -- this is what validates your pipeline
    DESIGN, it is not the path used for the exam dataset itself.
    """

    folds = []

    for fold in range(1, n_folds + 1):

        logger.info(f"Loading fold {fold}")

        dataset = Dataset.load(
            datadir=datadir,
            task=task,
            fold=fold
        )

        folds.append(
            (
                dataset.X_train,
                dataset.y_train,
                dataset.X_test,
                dataset.y_test
            )
        )

    logger.info("Running multi-fold-averaging HPO across all outer folds")

    automl = AutoML(seed=seed)
    automl.fit_multi_fold(folds)

    scores = automl.evaluate_per_fold(folds)

    # Save the last fold's predictions as a sanity-check artifact
    # (evaluate_per_fold already stores one model per fold in
    # automl.models, in the same order as `folds`).
    if output_path is not None:
        last_preds = automl.models[-1].predict(folds[-1][2])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            np.save(f, last_preds)

    logger.info(
        f"\n{task}: mean R\u00b2 = {scores.mean():.4f} "
        f"+/- {scores.std():.4f} over {n_folds} folds"
    )

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1", "exam_dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/bike_sharing_demand/1/predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
            " You are free to also evaluate on other folds for your own analysis."
            " For the exam dataset, there exists only a single fold, fold 1."
        ),
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=(
            "If set, runs evaluate_across_folds instead of main -- loops folds 1..n_folds"
            " and reports mean/std R^2. Only valid for practice datasets"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )

    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=(
            "The directory where the datasets are stored."
            " You should be able to mostly leave this as the default."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    if args.n_folds is not None:

        evaluate_across_folds(
            task=args.task,
            n_folds=args.n_folds,
            seed=args.seed,
            datadir=args.datadir,
            output_path=args.output_path,
        )
    else:
        main(
            task=args.task,
            fold=args.fold,
            output_path=args.output_path,
            datadir=args.datadir,
            seed=args.seed,
        )