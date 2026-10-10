"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

For the practice datasets you are given access to y_test; however,
for the exam dataset, you will not have access to these labels.
You will need to output your predictions for X_test to a file,
which we will grade using GitHub Classroom.

Two usage modes:

1) Single fold (original behaviour) -- writes predictions straight
   to --output-path:

    python run.py --task wine_quality --fold 3 --seed 42 --output-path demo_wine_quality.npy

2) All folds (new) -- runs every fold from 1..--n-folds, writes one
   predictions file per fold (suffixed with the fold number), and
   reports the mean +/- std R^2 across folds:

    python run.py --task wine_quality --n-folds 10 --seed 42 --output-path demo_wine_quality.npy
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")
from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
reexec_with_fixed_hash_seed()
pin_thread_env()

from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from automl.data import Dataset
from automl.core import AutoML
import argparse

import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def _fold_output_path(output_path: Path, fold: int) -> Path:
    """Insert '_fold{N}' before the file extension, e.g.
    'demo_wine_quality.npy' -> 'demo_wine_quality_fold3.npy'."""
    return output_path.with_name(f"{output_path.stem}_fold{fold}{output_path.suffix}")


def run_single_fold(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
) -> tuple[np.ndarray, float | None]:
    """Fits AutoML on one fold, writes predictions to output_path, and
    returns (test_preds, r2_test_or_None)."""
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info(f"Fitting AutoML [task={task}, fold={fold}]")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    automl = AutoML(seed=seed)
    automl.fit(dataset.X_train, dataset.y_train)
    test_preds: np.ndarray = automl.predict(dataset.X_test)

    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    logger.info(f"Writing predictions to disk: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    r2_test = None
    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"[fold {fold}] R^2 on test set: {r2_test:.4f}")
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info(f"[fold {fold}] No test labels (y_test) for task '{task}'")

    return test_preds, r2_test


def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    n_folds: int | None = None,
):
    # ------------------------------------------------------------------
    # Single-fold mode (original behaviour): unchanged, writes straight
    # to output_path.
    # ------------------------------------------------------------------
    if n_folds is None:
        run_single_fold(
            task=task, fold=fold, output_path=output_path, seed=seed, datadir=datadir,
        )
        return

    # ------------------------------------------------------------------
    # All-folds mode: loop folds 1..n_folds, one predictions file per
    # fold, then report mean +/- std R^2 across folds.
    # ------------------------------------------------------------------
    if task == "exam_dataset" and n_folds > 1:
        logger.warning(
            "'exam_dataset' only has a single fold (fold 1); "
            "ignoring --n-folds and running fold 1 only."
        )
        n_folds = 1

    r2_scores = []
    for fold_i in range(1, n_folds + 1):
        fold_output_path = _fold_output_path(output_path, fold_i)
        _, r2_test = run_single_fold(
            task=task, fold=fold_i, output_path=fold_output_path, seed=seed, datadir=datadir,
        )
        if r2_test is not None:
            r2_scores.append(r2_test)

    if r2_scores:
        r2_scores = np.array(r2_scores)
        mean_r2 = r2_scores.mean()
        std_r2 = r2_scores.std()
        logger.info(
            f"[{task}] R^2 across {len(r2_scores)}/{n_folds} folds: "
            f"{mean_r2:.4f} \u00b1 {std_r2:.4f}"
        )
        print(
            f"[{task}] Average TEST R2 across {len(r2_scores)} folds "
            f"(seed={seed}): {mean_r2:.4f} \u00b1 {std_r2:.4f}"
        )
        summary_path = output_path.with_name(f"{output_path.stem}_r2_summary.npy")
        np.save(summary_path, r2_scores)
        logger.info(f"Saved per-fold R^2 scores to {summary_path}")
    else:
        logger.info(
            f"[{task}] No y_test available for any fold -- nothing to average "
            f"(expected for 'exam_dataset')."
        )


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
            " In --n-folds mode, this is used as a base name: each fold's"
            " predictions are saved as '<stem>_fold<N><suffix>', and the"
            " per-fold R^2 scores are saved as '<stem>_r2_summary.npy'."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on (single-fold mode, ignored if --n-folds is set)."
            " You are free to also evaluate on other folds for your own analysis."
            " For the exam dataset, there exists only a single fold, fold 1."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=(
            "If set, run on folds 1..N instead of a single fold, and report"
            " the mean +/- std R^2 across all folds that have test labels."
            " Practice datasets have 10 folds available; 'exam_dataset' has"
            " only 1 fold regardless of this argument."
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

    main(
        task=args.task,
        fold=args.fold,
        output_path=args.output_path,
        datadir=args.datadir,
        seed=args.seed,
        n_folds=args.n_folds,
    )