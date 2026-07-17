# utils.py
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np


# Flags that make LightGBM bit-for-bit reproducible across runs/processes.
# random_state alone is NOT sufficient: by default LightGBM builds
# histograms across multiple threads and sums partial results in an order
# that depends on OS thread scheduling. Since floating-point addition is
# not associative, this produces tiny numerical differences run-to-run,
# which can snowball into different split decisions, different trees, and
# ultimately different R^2 -- and, since BOHB is fed this non-reproducible
# objective, potentially a different incumbent config too.
#
#   deterministic=True   -> forces LightGBM's deterministic code path
#   force_row_wise=True  -> avoids the multi-threaded histogram-building
#                            code path that is the actual source of the
#                            nondeterminism (this is the one that matters
#                            most; deterministic=True alone still allows
#                            some col-wise multithreaded accumulation)
#   n_jobs=1              -> removes thread-scheduling variance entirely
#                            (also required for the above two flags to be
#                            fully effective; -1/multi-thread + deterministic
#                            flags can still show tiny drift on some builds)
LGBM_DETERMINISM_KWARGS = dict(
    deterministic=True,
    force_row_wise=True,
    n_jobs=1,
)


def _fit_and_score(X_tr, y_tr, X_val, y_val, params, seed,
                    max_estimators=2000, early_stopping_rounds=50):
    """Shared low-level routine: train with early stopping, score on the
    given validation slice. Used only for the FINAL model refit (post-HPO),
    not during the BOHB search itself -- see internal_cv_score_budgeted
    below for why the search uses a fixed budget instead."""

    model = lgb.LGBMRegressor(
        random_state=seed,
        n_estimators=max_estimators,
        verbosity=-1,
        #**LGBM_DETERMINISM_KWARGS,
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


def internal_cv_score_budgeted(
    X_train,
    y_train,
    params,
    budget,
    seed,
    n_splits=5,
):
    """
    BOHB HPO objective (replaces internal_cv_score for the search itself).

    `budget` = n_estimators (boosting rounds) for THIS trial, chosen by
    SMAC's Hyperband intensifier -- low-budget trials screen many configs
    cheaply, only promising ones get re-evaluated at higher budgets.

    Deliberately NO early stopping here: the whole point of Hyperband is
    that the budget IS the fidelity dimension being controlled externally.
    If LightGBM's early stopping were also active, a "budget=50" trial and
    a "budget=1000" trial could converge to the same effective round count,
    silently collapsing the fidelity signal Hyperband relies on to
    distinguish cheap screening evals from expensive full evals.

    Given a single (X_train, y_train), builds its own K internal folds
    (never touching X_test), trains each fold at the fixed `budget`, and
    averages R2 across them -- same "every fold is its own exam dataset"
    principle as before.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    scores = []

    for train_idx, val_idx in kf.split(X_reset):
        X_tr, X_val = X_reset.iloc[train_idx], X_reset.iloc[val_idx]
        y_tr, y_val = y_reset.iloc[train_idx], y_reset.iloc[val_idx]

        model = lgb.LGBMRegressor(
            random_state=seed,
            n_estimators=int(budget),
            verbosity=-1,
            #**LGBM_DETERMINISM_KWARGS,
            **params,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(r2_score(y_val, preds))

    return float(np.mean(scores))


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
    Kept for backward compatibility / non-BOHB flat-BO comparisons.
    Not used by the BOHB path (see internal_cv_score_budgeted above).
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
    UNCHANGED from before -- runs AFTER the BOHB search has picked
    learning_rate/num_leaves/max_depth/min_child_samples. Determines a
    good number of boosting rounds via early stopping on a carved-out
    validation slice, then refits on ALL of X_train using that fixed
    round count. This is intentionally separate from the BOHB search's
    own budget mechanism: BOHB's `budget` decides how much compute a
    *config* deserves during the search; this decides the final deployed
    model's actual round count once the winning config is known.
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
        #**LGBM_DETERMINISM_KWARGS,
        **params,
    )
    final_model.fit(X_train, y_train)

    return final_model, n_rounds