# utils.py
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def _fit_and_score(X_tr, y_tr, X_val, y_val, params, seed,
                    max_estimators=2000, early_stopping_rounds=50):
    """Shared low-level routine: train with early stopping, score on the
    given validation slice. Used by both HPO tiers below so the actual
    model-fitting logic only lives in one place."""

    model = lgb.LGBMRegressor(
        random_state=seed,
        n_estimators=max_estimators,
        verbosity=-1,
        **params,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)

    return r2_score(y_val, preds), (model.best_iteration_ or max_estimators)


def internal_cv_score(
    X_train,
    y_train,
    params,
    seed,
    n_splits=5,
    max_estimators=2000,
    early_stopping_rounds=50,
):
    """
    TIER 1 -- Internal-CV HPO objective.

    Use this whenever you only have ONE (X_train, y_train) available --
    this is exactly the situation for the exam dataset, which has a
    single fold. We build our own K internal folds out of X_train (never
    touching X_test), train+early-stop on each, and average R2 across
    them. This function is fully self-contained: it never assumes outer
    folds exist, so it's identical code whether you call it on practice
    fold 3 or on the exam dataset.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    scores = []

    for train_idx, val_idx in kf.split(X_reset):
        X_tr, X_val = X_reset.iloc[train_idx], X_reset.iloc[val_idx]
        y_tr, y_val = y_reset.iloc[train_idx], y_reset.iloc[val_idx]

        score, _ = _fit_and_score(
            X_tr, y_tr, X_val, y_val, params, seed,
            max_estimators, early_stopping_rounds,
        )
        scores.append(score)

    return float(np.mean(scores))


def multi_fold_score(
    folds,
    params,
    seed,
    val_fraction=0.2,
    max_estimators=2000,
    early_stopping_rounds=50,
):
    """
    TIER 2 -- Multi-fold-averaging HPO objective.

    Use this ONLY when you have several OUTER folds available (the 10
    practice folds) and you want to check that a hyperparameter choice
    generalizes across different train/test splits of the same dataset
    -- not just within one of them. This is for validating your pipeline
    DESIGN on the practice data; it is NOT what you run on the exam
    dataset, since the exam dataset never gives you multiple folds.

    For each outer fold, carve a single validation slice out of that
    fold's own X_train (the fold's true X_test is never touched here),
    and average R2 across the outer folds.
    """

    scores = []

    for X_train, y_train, X_test, y_test in folds:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction, random_state=seed,
        )

        score, _ = _fit_and_score(
            X_tr, y_tr, X_val, y_val, params, seed,
            max_estimators, early_stopping_rounds,
        )
        scores.append(score)

    return float(np.mean(scores))


def fit_lgbm_with_early_stopping(
    X_train,
    y_train,
    params,
    seed,
    val_fraction=0.15,
    max_estimators=2000,
    early_stopping_rounds=50,
):
    """
    Determines a good number of boosting rounds via early stopping on a
    carved-out validation slice, then refits on ALL of X_train using that
    fixed round count -- so the deployed model still sees 100% of the
    available training data (rather than permanently losing val_fraction
    of it), while still being protected from overfitting.

    Used to produce the actual deployable model AFTER HPO (either tier)
    has already picked best_params.
    """

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_fraction, random_state=seed,
    )

    _, n_rounds = _fit_and_score(
        X_fit, y_fit, X_val, y_val, params, seed,
        max_estimators, early_stopping_rounds,
    )

    final_model = lgb.LGBMRegressor(
        random_state=seed,
        n_estimators=n_rounds,
        verbosity=-1,
        **params,
    )
    final_model.fit(X_train, y_train)

    return final_model, n_rounds