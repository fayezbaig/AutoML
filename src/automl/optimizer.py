# optimizer.py
from __future__ import annotations

from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import EI

from .search_space import get_configspace


class BayesianOptimizer:
    """
    SMAC3-backed replacement for the manual (surrogate.py + acquisition.py
    + hand-rolled loop) optimizer. Keeps the EXACT SAME external interface
    as before, so automl.py needs no changes:

        optimizer = BayesianOptimizer(seed, n_init, n_iter)
        best_params, best_score = optimizer.optimize(objective_fn)

    where objective_fn: callable(params: dict) -> float (higher is better).

    Internals:
        Surrogate model : Random Forest (HyperparameterOptimizationFacade's
                           default -- not overridden, replaces surrogate.py)
        Acquisition fn   : Expected Improvement, set explicitly (replaces
                           acquisition.py); SMAC3 defaults the facade to
                           LogEI otherwise.
        Initial design   : n_init random configurations (replaces the
                           manual random-search warmup loop)
        BO iterations    : n_iter model-guided evaluations
    """

    def __init__(self, seed: int, n_init: int = 10, n_iter: int = 30):
        self.seed = seed
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_trials = n_init + n_iter

    def optimize(self, objective_fn):
        cs = get_configspace(seed=self.seed)

        def target_function(config, seed: int = 0) -> float:
            params = dict(config)
            score = objective_fn(params)
            return -score  # SMAC minimizes -> flip sign of the R^2-based score

        scenario = Scenario(
            cs,
            n_trials=self.n_trials,
            deterministic=True,
            seed=self.seed,
        )

        initial_design = HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=self.n_init
        )

        acquisition_function = EI()

        model = HyperparameterOptimizationFacade.get_model(scenario, n_trees=200)

        smac = HyperparameterOptimizationFacade(
            scenario,
            target_function,
            model=model,
            initial_design=initial_design,
            acquisition_function=acquisition_function,
            overwrite=True,
        )

        incumbent = smac.optimize()
        best_loss = smac.validate(incumbent)

        best_params = dict(incumbent)
        best_score = -best_loss

        return best_params, best_score