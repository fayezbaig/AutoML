"""Test a classic real-world gotcha: a categorical value that appears
in X_test but was NEVER seen during training. Common failure mode for
naive encoders."""
import numpy as np
import pandas as pd
from automl.core import AutoML
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)

X_train = pd.DataFrame({
    "num": rng.normal(size=300),
    "cat": rng.choice(["A", "B", "C"], size=300),  # only A, B, C seen in training
})
y_train = pd.Series(rng.normal(size=300))

X_test = pd.DataFrame({
    "num": rng.normal(size=100),
    "cat": rng.choice(["A", "B", "C", "D", "E"], size=100),  # D, E are UNSEEN
})
y_test = pd.Series(rng.normal(size=100))

model = AutoML(seed=0, portfolio=("lightgbm", "random_forest", "catboost"), use_tabpfn=False)
model.fit(X_train, y_train)

try:
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"SUCCESS -- R2 on held-out test (with unseen categories D, E): {score:.4f}")
except Exception as e:
    print(f"FAILED -- unseen category caused a crash: {e}")
    raise
