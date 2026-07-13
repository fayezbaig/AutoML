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
    Single-fold path -- this is exactly what will run on the exam
    dataset: one (X_train, y_train, X_test), one independent internal-CV
    HPO search (SMAC BO with RF surrogate + EI acquisition), one final
    model.
    """
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Fitting AutoML (internal-CV HPO via SMAC)")

    automl = AutoML(seed=seed)
    automl.fit(dataset.X_train, dataset.y_train)
    test_preds: np.ndarray = automl.predict(dataset.X_test)

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
    Multi-fold validation path -- only valid for practice datasets.

    Runs the SAME per-fold-independent HPO procedure that will run on
    the exam dataset, once per fold, with no sharing of hyperparameters
    or data between folds. Reports the mean/std of the resulting honest
    test R2 across folds, plus how much the independently-chosen
    hyperparameters agree with each other.
    """

    folds = []

    for fold in range(1, n_folds + 1):
        logger.info(f"Loading fold {fold}")
        dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
        folds.append(
            (dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
        )

    logger.info("Running per-fold-independent HPO across all outer folds")

    scores, params_list, models = AutoML.run_per_fold(folds, seed=seed)

    # Save the last fold's predictions as a sanity-check artifact
    if output_path is not None:
        last_preds = models[-1].predict(folds[-1][2])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            np.save(f, last_preds)

    logger.info(
        f"\n{task}: mean R\u00b2 = {scores.mean():.4f} "
        f"+/- {scores.std():.4f} over {n_folds} folds"
    )

    return scores, params_list


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
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on. For the exam dataset, there exists only"
            " a single fold, fold 1."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=(
            "If set, runs evaluate_across_folds instead of main -- loops"
            " folds 1..n_folds, each with its own independent HPO search,"
            " and reports mean/std R^2. Only valid for practice datasets."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datadir", type=Path, default=DATADIR)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(f"Running task {args.task}\n{args}")

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