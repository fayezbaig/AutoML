"""
preprocessing.py — SINGLE SOURCE OF TRUTH for data preprocessing, used
by every script in this project. Previously encode_categoricals() was
defined once in optimizer_comparison.py and imported everywhere else,
while preprocess_for_algo() (CatBoost's raw-categorical handling) lived
separately in extended_portfolio.py -- functionally fine, but the two
related preprocessing concerns were split across files with an unclear
relationship. Consolidating them here makes the "what preprocessing
does each algorithm get" story explicit in one place.
"""


def encode_categoricals(X):
    """Label-encode object/category dtype columns -- used by LightGBM,
    RandomForest, XGBoost, ElasticNet (with StandardScaler added on top
    for ElasticNet specifically, see preprocess_for_algo)."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name in ("object", "category"):
            X[col] = X[col].astype("category").cat.codes
    return X


def preprocess_for_algo(algo_name, X_train, X_test):
    """(See docstring below.) Missing values are imputed FIRST, before
    any algorithm-specific encoding, so every downstream path receives
    clean data -- our practice datasets happened to have zero missing
    values, but this cannot be assumed for the exam dataset."""
    X_train, X_test = handle_missing_values(X_train, X_test)
    return _preprocess_for_algo_inner(algo_name, X_train, X_test)


def _preprocess_for_algo_inner(algo_name, X_train, X_test):
    """Returns (X_train_processed, X_test_processed, cat_features_idx).
    cat_features_idx is only meaningful for CatBoost (None otherwise).

    Per-algorithm rationale:
      - catboost: RAW categorical columns -- CatBoost's native categorical
        handling (ordered target statistics) is its key differentiator;
        label-encoding away would defeat the purpose of including it.
      - elasticnet: label-encoded + StandardScaler -- linear models are
        sensitive to feature scale, unlike tree-based methods.
      - tabpfn: RAW, UNSCALED, UNENCODED -- per TabPFN's own documentation,
        scaling/encoding is unnecessary and not effective. NOTE: tabpfn
        preprocessing is handled directly in ensemble.py (it needs the
        raw train/val/test splits, not a train/test pair), not here.
      - everything else (lightgbm, random_forest, xgboost): label-encoded
        only, no scaling (tree splits are scale-invariant).
    """
    if algo_name == "catboost":
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = [
            c for c in X_tr.columns if X_tr[c].dtype.name in (
                "object", "category")]
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype(str)
            X_te[c] = X_te[c].astype(str)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        return X_tr, X_te, cat_idx

    X_tr = encode_categoricals(X_train)
    X_te = encode_categoricals(X_test)

    if algo_name == "elasticnet":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        return X_tr_scaled, X_te_scaled, None

    return X_tr, X_te, None


class MissingValueImputer:
    """
    Fitted imputer: LEARNS its decisions (which columns get a
    "_was_missing" indicator, what fill value to use) ONCE from training
    data via fit(), then applies those EXACT SAME decisions consistently
    at transform() time -- regardless of the missingness pattern in
    whatever data is being transformed.

    This fixes a real bug: the original stateless handle_missing_values()
    recomputed "which columns need indicators" fresh on every call,
    including at predict() time using only the (possibly differently-
    distributed) test/inference data -- causing a train/predict feature
    count mismatch (e.g. 86 features at train time vs 91 at predict
    time) whenever a column's missingness happened to cross the 5%
    threshold on one split but not the other. Observed on the Ames
    housing dataset (severe missingness: PoolQC 99.5%, MiscFeature
    96.3%), which exposed this edge case that Titanic/Adult census
    (milder missingness) did not.
    """

    def __init__(self, missing_threshold=0.05):
        self.missing_threshold = missing_threshold
        self.add_indicator_ = {}   # col -> bool, decided ONCE at fit time
        self.fill_values_ = {}     # col -> fill value, decided ONCE at fit time
        self.columns_ = None

    def fit(self, X_train):
        self.columns_ = list(X_train.columns)
        for col in self.columns_:
            n_missing = X_train[col].isna().sum()
            missing_frac = n_missing / len(X_train) if len(X_train) > 0 else 0
            self.add_indicator_[col] = missing_frac > self.missing_threshold

            if X_train[col].dtype.name in ("object", "category"):
                mode = X_train[col].mode(dropna=True)
                self.fill_values_[col] = mode.iloc[0] if len(
                    mode) > 0 else "missing"
            else:
                self.fill_values_[col] = X_train[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_:
            if col not in X.columns:
                continue
            if self.add_indicator_[col]:
                X[f"{col}_was_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(self.fill_values_[col])
        return X

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)


def handle_missing_values(X_train, X_test):
    """
    Convenience wrapper: fit the imputer on X_train ONLY, apply the
    SAME learned decisions to both X_train and X_test. For repeated
    use (e.g. predict() called many times after one fit()), prefer
    fitting a MissingValueImputer once and reusing it directly, rather
    than calling this function repeatedly.
    """
    imputer = MissingValueImputer()
    imputer.fit(X_train)
    X_train_clean = imputer.transform(X_train)
    X_test_clean = imputer.transform(X_test)
    return X_train_clean, X_test_clean


class FeatureEngineer:
    """
    Generic feature engineering for anonymized tabular data (no semantic
    column meaning available, so only domain-agnostic techniques apply).
    Adds pairwise interaction terms (products) for NUMERIC columns only,
    following the same fit-once/transform-consistently pattern as
    MissingValueImputer -- which numeric columns to use is decided ONCE
    from training data, then applied identically at transform time.

    Capped at max_numeric_cols columns (by variance, highest first) to
    avoid combinatorial explosion on wide datasets -- interaction terms
    grow as O(n^2) in the number of numeric columns.
    """

    def __init__(self, max_numeric_cols=10, degree=2):
        self.max_numeric_cols = max_numeric_cols
        self.degree = degree
        self.selected_cols_ = None
        self.poly_ = None

    def fit(self, X_train):
        numeric_cols = [
            c for c in X_train.columns if X_train[c].dtype.name not in (
                "object", "category")]
        if len(numeric_cols) == 0:
            self.selected_cols_ = []
            return self

        if len(numeric_cols) > self.max_numeric_cols:
            variances = X_train[numeric_cols].var(
            ).sort_values(ascending=False)
            numeric_cols = list(variances.index[:self.max_numeric_cols])

        self.selected_cols_ = numeric_cols

        if len(numeric_cols) >= 2:
            from sklearn.preprocessing import PolynomialFeatures
            self.poly_ = PolynomialFeatures(
                degree=self.degree,
                interaction_only=True,
                include_bias=False)
            self.poly_.fit(X_train[numeric_cols].fillna(0))
        return self

    def transform(self, X):
        if not self.selected_cols_ or self.poly_ is None:
            return X.copy()

        X = X.copy()
        poly_features = self.poly_.transform(X[self.selected_cols_].fillna(0))
        n_original = len(self.selected_cols_)
        # skip the original columns, keep only NEW interaction terms
        interaction_features = poly_features[:, n_original:]

        feature_names = self.poly_.get_feature_names_out(self.selected_cols_)[
            n_original:]
        for i, name in enumerate(feature_names):
            safe_name = f"interact_{name.replace(' ', '_x_')}"
            X[safe_name] = interaction_features[:, i]

        return X

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)


def check_tabpfn_available(timeout_seconds=120):
    """Fast pre-flight check: is TabPFN genuinely usable right now (import
    succeeds, token/license valid, a real tiny fit()+predict() works)?
    Returns (available: bool, reason: str). Meant to be called ONCE,
    cheaply, before committing to a full multi-algorithm search -- so a
    broken TabPFN setup (missing token, invalid license, no GPU when
    required, etc.) fails fast with a clear reason, rather than being
    silently discovered deep inside a multi-hour fit() call.

    Runs the check in a background thread with a timeout, since a
    missing token can trigger TabPFN's interactive browser-login flow,
    which would otherwise block indefinitely. If a human is available
    to complete the login within timeout_seconds, TabPFN becomes
    available for the rest of this run; otherwise we fall back
    gracefully to the tree-based portfolio. Thread-based (not
    signal-based) so this works identically on Windows, macOS, Linux."""
    import threading

    result = {
        "available": False,
        "reason": f"timed out after {timeout_seconds}s "
        f"(likely waiting on an interactive login prompt)"}

    def _check():
        try:
            import numpy as np
            import pandas as pd
            from tabpfn import TabPFNRegressor

            X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)))
            y = pd.Series(np.random.default_rng(0).normal(size=20))

            model = TabPFNRegressor(device="cpu")
            model.fit(X, y)
            _ = model.predict(X)
            result["available"] = True
            result["reason"] = "OK"
        except Exception as e:
            result["reason"] = str(e)

    thread = threading.Thread(target=_check, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    return result["available"], result["reason"]


def validate_no_duplicate_columns(X):
    """Fail fast and clearly if X has duplicate column names, rather
    than letting downstream preprocessing fail with a confusing,
    unrelated-looking error (e.g. 'DataFrame has no attribute dtype'
    from encode_categoricals, caused by X[col] returning a DataFrame
    instead of a Series when col appears more than once)."""
    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].unique().tolist()
        raise ValueError(
            f"Duplicate column names found: {dupes}. AutoML requires "
            f"uniquely-named columns; please rename or drop duplicates "
            f"before calling fit()."
        )
