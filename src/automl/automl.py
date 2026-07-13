# automl.py
from __future__ import annotations

from functools import partial
import logging

import numpy as np
from sklearn.metrics import r2_score

from .optimizer import BayesianOptimizer
from .utils import internal_cv_score, fit_lgbm_with_early_stopping


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int, n_init: int = 10, n_iter: int = 30):
        self.seed = seed
        self.n_init = n_init
        self.n_iter = n_iter

        self.best_params = None
        self.best_cv_score = None
        self._final_model = None

    # ------------------------------------------------------------------
    # Core HPO + fit for ONE dataset (one fold's worth of data). This is
    # the exact method that will run on the exam dataset -- internal
    # K-fold CV on X_train picks the hyperparameters (now via SMAC BO
    # with a Random Forest surrogate + EI acquisition), then a final
    # model is refit on the full X_train.
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, n_splits: int = 5):

        objective = partial(
            internal_cv_score,
            X_train,
            y_train,
            seed=self.seed,
            n_splits=n_splits,
        )

        optimizer = BayesianOptimizer(
            seed=self.seed, n_init=self.n_init, n_iter=self.n_iter
        )
        best_params, best_score = optimizer.optimize(objective)

        logger.info(f"Best internal-CV R2: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        self.best_params = best_params
        self.best_cv_score = best_score

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
    # own internal-CV HPO search, its own best_params, its own model --
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
    def run_per_fold(folds, seed, n_init: int = 10, n_iter: int = 30, n_splits: int = 5):

        scores = []
        params_list = []
        models = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(folds):

            logger.info(f"--- Fold {i+1}/{len(folds)}: independent HPO ---")

            automl = AutoML(seed=seed, n_init=n_init, n_iter=n_iter)
            automl.fit(X_train, y_train, n_splits=n_splits)

            preds = automl.predict(X_test)
            score = r2_score(y_test, preds)

            scores.append(score)
            params_list.append(automl.best_params)
            models.append(automl._final_model)

            logger.info(f"Fold {i+1} test R2: {score:.4f} | params: {automl.best_params}")

        scores = np.array(scores)
        logger.info(
            f"Per-fold independent HPO -- mean R2={scores.mean():.4f} "
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