# automl.py
from __future__ import annotations

from functools import partial
import logging

import numpy as np
from sklearn.metrics import r2_score

from .optimizer import BOHBOptimizer
from .utils import internal_cv_score_budgeted, fit_lgbm_with_early_stopping


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
        min_budget: int = 50,
        max_budget: int = 300,
        n_trials: int = 60,
        eta: int = 3,
        n_init: int = 10,
        walltime_limit: float | None = None,
        run_id: str | None = None,
    ):
        self.seed = seed
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.n_trials = n_trials
        self.eta = eta
        self.n_init = n_init
        self.walltime_limit = walltime_limit  # seconds; exam's 24h budget is 86400
        # Identifies this run's SMAC output directory (see optimizer.py /
        # determinism.py) so it never collides with another task/fold/seed's
        # run. Reproducibility across process reruns also depends on
        # PYTHONHASHSEED being pinned and n_workers=1 -- see
        # automl/determinism.py, which run.py / evaluate_folds.py apply.
        self.run_id = run_id

        self.best_params = None
        self.best_cv_score = None
        self._final_model = None

    # ------------------------------------------------------------------
    # Core HPO + fit for ONE dataset (one fold's worth of data). This is
    # the exact method that will run on the exam dataset -- internal
    # K-fold CV on X_train picks the hyperparameters via SMAC's BOHB
    # (Random Forest surrogate + EI acquisition, Hyperband budget
    # allocation over n_estimators), then a final model is refit on the
    # full X_train using early stopping to fix the deployed round count.
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, n_splits: int = 5):

        # objective(params, budget) -> avg CV R2 at that budget
        objective = partial(
            internal_cv_score_budgeted,
            X_train,
            y_train,
            seed=self.seed,
            n_splits=n_splits,
        )

        logger.info(
            f"CONFIG: trials={self.n_trials}, "
            f"init={self.n_init}, "
            f"min_budget={self.min_budget}, "
            f"max_budget={self.max_budget}, "
            f"cv={n_splits}"
    )

        optimizer = BOHBOptimizer(
            seed=self.seed,
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            n_trials=self.n_trials,
            eta=self.eta,
            n_init=self.n_init,
            walltime_limit=self.walltime_limit,
            run_id=self.run_id,
        )
        best_params, best_score = optimizer.optimize(objective)

        logger.info(f"Best internal-CV R2 (BOHB incumbent): {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        self.best_params = best_params
        self.best_cv_score = best_score

        # Final deployed model: n_estimators is picked here via early
        # stopping, independently of BOHB's fidelity budget above.
        self._final_model, n_rounds = fit_lgbm_with_early_stopping(
            X_train, y_train, self.best_params, self.seed
        )
        logger.info(f"Final model trained with n_rounds={n_rounds}")

        return self

    def fit_final_model(self, X_train, y_train):
        """Retrain a deployable model on given data using self.best_params
        (useful if you want to pool data after fit() already ran once)."""

        if self.best_params is None:
            raise ValueError("Must call fit() first to determine best_params.")

        model, n_rounds = fit_lgbm_with_early_stopping(
            X_train, y_train, self.best_params, self.seed
        )
        logger.info(f"Final model trained with n_rounds={n_rounds}")

        self._final_model = model
        return self

    def predict(self, X):
        if self._final_model is None:
            raise ValueError("No final model available. Call fit() first.")
        return self._final_model.predict(X)

    # ------------------------------------------------------------------
    # Automated per-fold-independent runner.
    #
    # For EACH fold, runs a completely fresh, independent fit() -- its
    # own internal-CV BOHB search, its own best_params, its own model --
    # exactly simulating "if this fold were the exam dataset." No fold's
    # search ever sees another fold's data, and no hyperparameters are
    # shared or averaged across folds during the search itself.
    #
    # At the end, the resulting R2 scores (and the chosen best_params)
    # are collected across folds so you can look at the MEAN/STD of
    # outcomes -- and separately, how much the chosen hyperparameters
    # agree with each other -- without ever mixing folds during training.
    # ------------------------------------------------------------------
    @staticmethod
    def run_per_fold(
        folds,
        seed,
        min_budget: int = 50,
        max_budget: int = 300,
        n_trials: int = 60,
        eta: int = 3,
        n_init: int = 10,
        n_splits: int = 5,
        walltime_limit: float | None = None,
        run_id_prefix: str = "run",
    ):

        scores = []
        params_list = []
        models = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(folds):

            logger.info(f"--- Fold {i+1}/{len(folds)}: independent BOHB HPO ---")

            automl = AutoML(
                seed=seed,
                min_budget=min_budget,
                max_budget=max_budget,
                n_trials=n_trials,
                eta=eta,
                n_init=n_init,
                walltime_limit=walltime_limit,
                # Distinct per-fold run_id -- see optimizer.py. Without
                # this, every fold reuses the same seed + identical
                # scenario settings, so SMAC would hash all folds to the
                # SAME smac3_output/<name>/<seed> directory.
                run_id=f"{run_id_prefix}_fold{i+1}",
            )
            automl.fit(X_train, y_train, n_splits=n_splits)

            preds = automl.predict(X_test)
            score = r2_score(y_test, preds)

            scores.append(score)
            params_list.append(automl.best_params)
            models.append(automl._final_model)

            logger.info(f"Fold {i+1} test R2: {score:.4f} | params: {automl.best_params}")

        scores = np.array(scores)
        logger.info(
            f"Per-fold independent BOHB HPO -- mean R2={scores.mean():.4f} "
            f"+/- {scores.std():.4f} over {len(folds)} folds"
        )

        AutoML._summarize_param_spread(params_list)

        return scores, params_list, models

    @staticmethod
    def _summarize_param_spread(params_list):
        """Logs how much the independently-chosen hyperparameters agree
        across folds -- convergent values are reassuring evidence the
        search space/method is sound; scattered values are worth flagging
        as a limitation."""

        if not params_list:
            return

        for key in params_list[0].keys():
            values = np.array([p[key] for p in params_list])
            logger.info(
                f"  {key}: mean={values.mean():.4g}, std={values.std():.4g}, "
                f"min={values.min():.4g}, max={values.max():.4g}"
            )