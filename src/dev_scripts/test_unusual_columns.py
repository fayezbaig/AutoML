"""Test robustness against column types not seen in our 5 practice
datasets: boolean, all-constant (zero variance), and a datetime-like
numeric column. These could plausibly appear in the real exam dataset."""
import numpy as np
import pandas as pd
from automl.core import AutoML
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 500

X = pd.DataFrame({
    "normal_numeric": rng.normal(size=n),
    "bool_col": rng.choice([True, False], size=n),
    "constant_col": np.ones(n),  # zero variance -- can break some scalers/encoders
    "categorical_col": rng.choice(["A", "B", "C"], size=n),
    "large_scale_numeric": rng.normal(loc=1e6, scale=1e5, size=n),  # datetime-like magnitude
})
y = pd.Series(rng.normal(size=n))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Column dtypes:")
print(X.dtypes)
print()

model = AutoML(seed=0, portfolio=("lightgbm", "random_forest", "catboost"), use_tabpfn=False)
model.fit(X_train, y_train)
preds = model.predict(X_test)
score = r2_score(y_test, preds)

print(f"SUCCESS -- R2 on held-out test: {score:.4f}")
print(f"Deployment mode: {model.deployment_mode_}")
