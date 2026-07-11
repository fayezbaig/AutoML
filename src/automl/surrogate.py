from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RandomForestSurrogate:

    def __init__(self, seed=42):

        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=seed
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, x):

        x = np.asarray(x).reshape(1, -1)

        tree_predictions = np.array([
            tree.predict(x)[0]
            for tree in self.model.estimators_
        ])

        mean = tree_predictions.mean()

        std = tree_predictions.std()

        return mean, std