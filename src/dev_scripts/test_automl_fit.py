from pathlib import Path
from automl.data import Dataset
from automl.core import AutoML
from sklearn.metrics import r2_score

ds = Dataset.load(datadir=Path('../data'), task='wine_quality', fold=1)
model = AutoML(seed=0).fit(ds.X_train, ds.y_train)
preds = model.predict(ds.X_test)
print('R2:', r2_score(ds.y_test, preds))
print('Ensemble composition:', model.ensemble_composition_)
