import numpy as np


class SearchSpace:

    def __init__(self, seed=42):

        self.rng = np.random.RandomState(seed)

    def sample(self):

        return {
            "learning_rate": self.rng.uniform(0.01, 0.20),
            "num_leaves": self.rng.randint(20, 100)
        }

    def to_array(self, config):

        return np.array([
            config["learning_rate"],
            config["num_leaves"]
        ])

    def from_array(self, x):

        return {

            "learning_rate": float(x[0]),

            "num_leaves": int(round(x[1]))
        }