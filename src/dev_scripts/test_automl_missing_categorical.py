"""Second missing-value robustness test -- 'adult' census dataset.
Different pattern than Titanic: missingness concentrated entirely in
CATEGORICAL columns (workclass, occupation, native-country), at larger
scale (~2800 missing per column, ~48k total rows)."""
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from automl.core import AutoML

dataset = openml.datasets.get_dataset(1590)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# 'adult' is normally a classification dataset (income >50K or <=50K);
# use 'age' as a numeric regression target instead, dropping it from X.
y = X["age"].copy()
X = X.drop(columns=["age"])

print(f"X shape: {X.shape}, missing values in X: {X.isna().sum().sum()}")
print(f"y (target) missing values: {y.isna().sum()} (should be 0)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Fitting AutoML on data with categorical-concentrated missing values...")
model = AutoML(seed=0).fit(X_train, y_train)
preds = model.predict(X_test)

score = r2_score(y_test, preds)
print(f"SUCCESS -- R2 on held-out test: {score:.4f}")
print(f"Ensemble composition: {model.ensemble_composition_}")
