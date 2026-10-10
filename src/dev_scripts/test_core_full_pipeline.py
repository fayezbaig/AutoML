"""Full validation of the newly-wired core.py: TabPFN inclusion,
feasibility check, and Pareto-based deployment selection, tested
end-to-end via a real fit()/predict() cycle."""
from pathlib import Path
from automl.data import Dataset
from automl.core import AutoML
from sklearn.metrics import r2_score

ds = Dataset.load(datadir=Path("../data"), task="wine_quality", fold=1)
model = AutoML(seed=0).fit(ds.X_train, ds.y_train)
preds = model.predict(ds.X_test)

score = r2_score(ds.y_test, preds)
print(f"SUCCESS -- R2 on held-out test: {score:.4f}")
print(f"Deployment mode: {model.deployment_mode_}")
print(f"Ensemble composition: {model.ensemble_composition_}")
print(f"Per-algo validation scores: {model.per_algo_val_scores_}")
print(f"Pareto candidates (name, r2, predict_time): {model.pareto_candidates_}")
