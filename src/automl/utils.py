# utils.py
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def _fit_and_score(X_tr, y_tr, X_val, y_val, params, seed,
                    max_estimators=2000, early_stopping_rounds=50):
    """Shared low-level routine: train with early stopping, score on the
    given validation slice."""

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
    HPO objective used for EVERY fold, exam dataset included.

    Given a single (X_train, y_train), builds its own K internal folds
    (never touching X_test), trains+early-stops on each rotation, and
    averages R2 across them. This is the ONLY HPO objective in the
    pipeline now -- every fold, practice or exam, is treated as its own
    independent "exam dataset" and gets its own independent search.

    `params` arrives as a plain dict -- SMAC's optimizer.py already
    converts the Configuration object via dict(config) before calling
    this, so no changes were needed here.
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
    available training data, while still being protected from overfitting.
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