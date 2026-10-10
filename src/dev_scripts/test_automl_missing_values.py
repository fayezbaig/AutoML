"""Confirm the REAL AutoML.fit()/predict() pipeline handles missing
FEATURE values correctly. Uses Titanic (genuine missing values in X)
with 'age' as the target -- dropping any rows where the TARGET itself
is missing first (a real exam dataset guarantees a clean target;
only X may have gaps, which is what we're actually testing here)."""
import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from automl.core import AutoML

dataset = openml.datasets.get_dataset(40945)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Use 'age' as target (has missing values itself, so we must drop those
# rows -- mirrors how a real supervised dataset must have a clean target)
y = X["age"].copy()
X = X.drop(columns=["age"])

valid_target = ~y.isna()
X, y = X[valid_target], y[valid_target]

print(f"X shape: {X.shape}, missing values in X (features): {X.isna().sum().sum()}")
print(f"y (target) missing values: {y.isna().sum()} (should be 0)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Fitting AutoML on data WITH missing FEATURE values...")
model = AutoML(seed=0).fit(X_train, y_train)
preds = model.predict(X_test)

score = r2_score(y_test, preds)
print(f"SUCCESS -- R2 on held-out test (features had missing values): {score:.4f}")
print(f"Ensemble composition: {model.ensemble_composition_}")
