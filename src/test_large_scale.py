"""Stress test at genuinely larger scale than previous tests (California
housing ~20K, diamonds ~54K) -- pushing toward where TabPFN's feasibility
cutoff and overall pipeline runtime get meaningfully tested under load."""
import openml
from sklearn.model_selection import train_test_split
from automl.core import AutoML
from sklearn.metrics import r2_score

results = openml.datasets.list_datasets(data_name="black_friday", output_format="dataframe")
print(results[["did", "name", "NumberOfInstances", "NumberOfFeatures"]])

did = int(results.iloc[0]["did"])
dataset = openml.datasets.get_dataset(did)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

print(f"X shape: {X.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = AutoML(seed=0).fit(X_train, y_train)
preds = model.predict(X_test)
score = r2_score(y_test, preds)
print(f"SUCCESS -- R2 on held-out test: {score:.4f}")
print(f"Deployment mode: {model.deployment_mode_}")
print(f"Pareto candidates: {model.pareto_candidates_}")
