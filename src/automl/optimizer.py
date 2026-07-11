import numpy as np

from .surrogate import RandomForestSurrogate
from .acquisition import ExpectedImprovement
from .search_space import SearchSpace


class BayesianOptimizer:

    def __init__(
        self,
        seed,
        n_init=10,
        n_iter=30
    ):

        self.seed = seed

        self.n_init = n_init
        self.n_iter = n_iter

        self.space = SearchSpace(seed)

        self.surrogate = RandomForestSurrogate(seed)

        self.acquisition = ExpectedImprovement()

    def optimize(self, objective_fn):
        """
        objective_fn: callable(params: dict) -> float (higher is better)

        The optimizer itself is agnostic to WHAT the score means -- it's
        the caller's job to hand in the right objective, e.g.
        `partial(internal_cv_score, X_train, y_train, seed=seed)` for a
        single dataset (Tier 1), or `partial(multi_fold_score, folds,
        seed=seed)` for averaging over several outer folds (Tier 2).
        This keeps the search algorithm identical in both cases.
        """

        X_history = []
        y_history = []

        best_score = -np.inf
        best_params = None

        # Initial random search

        for _ in range(self.n_init):

            params = self.space.sample()

            score = objective_fn(params)

            X_history.append(
                self.space.to_array(params)
            )

            y_history.append(score)

            if score > best_score:

                best_score = score
                best_params = params

        # Bayesian optimization

        for _ in range(self.n_iter):

            self.surrogate.fit(
                np.array(X_history),
                np.array(y_history)
            )

            best_candidate = None
            best_ei = -np.inf

            for _ in range(100):

                candidate = self.space.sample()

                x = self.space.to_array(candidate)

                mean, std = self.surrogate.predict(x)

                ei = self.acquisition.compute(
                    mean,
                    std,
                    best_score
                )

                if ei > best_ei:

                    best_ei = ei
                    best_candidate = candidate

            score = objective_fn(best_candidate)

            X_history.append(
                self.space.to_array(best_candidate)
            )

            y_history.append(score)

            if score > best_score:

                best_score = score
                best_params = best_candidate

        return best_params, best_score