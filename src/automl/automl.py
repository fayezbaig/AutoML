# automl.py
from __future__ import annotations

from functools import partial
import logging

import numpy as np
from sklearn.metrics import r2_score

from .optimizer import BayesianOptimizer
from .utils import (
    internal_cv_score,
    multi_fold_score,
    fit_lgbm_with_early_stopping,
)


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int, n_init: int = 10, n_iter: int = 30):
        self.seed = seed
        self.n_init = n_init
        self.n_iter = n_iter

        self.models = []
        self.best_params = None
        self.best_cv_score = None
        self._final_model = None

    # ------------------------------------------------------------------
    # TIER 1: single dataset -> internal-CV HPO
    #
    # This is the method to use when you only have ONE (X_train, y_train)
    # -- which is exactly the exam dataset's situation (single fold).
    # It's also what run.py's `main()` calls, and it leaves you with a
    # deployable model immediately (predict() works right after fit()).
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

        logger.info(f"[internal-CV HPO] Best CV R2: {best_score:.4f}")
        logger.info(f"[internal-CV HPO] Best parameters: {best_params}")

        self.best_params = best_params
        self.best_cv_score = best_score

        self._final_model, n_rounds = fit_lgbm_with_early_stopping(
            X_train, y_train, self.best_params, self.seed
        )
        logger.info(f"Final model trained with n_rounds={n_rounds}")

        return self

    # ------------------------------------------------------------------
    # TIER 2: multiple outer folds -> multi-fold-averaging HPO
    #
    # Use this ONLY to validate your pipeline DESIGN across the 10
    # practice folds (does the same search space / model choice work
    # well across different train/test splits of the same dataset?).
    # This is NOT what you run on the exam dataset -- it only ever gives
    # you one fold, so fit() (Tier 1) is the only valid path there.
    # ------------------------------------------------------------------
    def fit_multi_fold(self, folds):

        objective = partial(multi_fold_score, folds, seed=self.seed)

        optimizer = BayesianOptimizer(
            seed=self.seed, n_init=self.n_init, n_iter=self.n_iter
        )
        best_params, best_score = optimizer.optimize(objective)

        logger.info(f"[multi-fold HPO] Best avg CV R2: {best_score:.4f}")
        logger.info(f"[multi-fold HPO] Best parameters: {best_params}")

        self.best_params = best_params
        self.best_cv_score = best_score

        return self

    def evaluate_per_fold(self, folds):
        """
        Honest per-fold evaluation: for each outer fold, train a fresh
        model on ONLY that fold's own training data (using whatever
        best_params were selected by fit_multi_fold), evaluate on ONLY
        that fold's own test data. This is what tells you if your
        hyperparameter choice is reliable across folds, not just on
        average.
        """

        if self.best_params is None:
            raise ValueError(
                "Must call fit() or fit_multi_fold() before evaluate_per_fold() "
                "to determine best_params."
            )

        scores = []
        self.models = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(folds):

            model, n_rounds = fit_lgbm_with_early_stopping(
                X_train, y_train, self.best_params, self.seed
            )
            self.models.append(model)

            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            scores.append(score)

            logger.info(f"Fold {i+1} R2: {score:.4f} (n_rounds={n_rounds})")

        scores = np.array(scores)
        logger.info(f"Per-fold mean R2={scores.mean():.4f} +/- {scores.std():.4f}")

        return scores

    def fit_final_model(self, X_train, y_train):
        """
        Trains a single deployable model on whatever data you pass in,
        using self.best_params + early stopping. For the exam dataset
        this already happens automatically inside fit() -- this method
        is only useful separately if you want to, e.g., pool several
        practice folds into one combined model for your own analysis.
        """

        if self.best_params is None:
            raise ValueError(
                "Must determine best_params first via fit() or fit_multi_fold()."
            )

        model, n_rounds = fit_lgbm_with_early_stopping(
            X_train, y_train, self.best_params, self.seed
        )
        logger.info(f"Final model trained with n_rounds={n_rounds}")

        self._final_model = model
        return self

    def predict(self, X):
        if self._final_model is None:
            raise ValueError(
                "No final model available. Call fit() (Tier 1) or "
                "fit_final_model() first."
            )
        return self._final_model.predict(X)