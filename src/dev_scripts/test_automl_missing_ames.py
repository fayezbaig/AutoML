"""Third missing-value robustness test -- Ames housing dataset. Genuinely
harder case than Titanic/adult: several columns are 90%+ missing
(PoolQC 99.5%, MiscFeature 96.3%, Alley 93.8%), stress-testing whether
mode-imputation + missingness-indicator columns hold up when a column
is almost entirely empty."""
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from automl.core import AutoML

dataset = openml.datasets.get_dataset(42165)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

print(f"X shape: {X.shape}, missing values in X: {X.isna().sum().sum()}")
print(f"y (target, SalePrice) missing values: {y.isna().sum()} (should be 0)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Fitting AutoML on data with SEVERE missing values (some columns >90% empty)...")
model = AutoML(seed=0).fit(X_train, y_train)
preds = model.predict(X_test)

score = r2_score(y_test, preds)
print(f"SUCCESS -- R2 on held-out test: {score:.4f}")
print(f"Ensemble composition: {model.ensemble_composition_}")
