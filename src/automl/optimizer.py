# optimizer.py
from __future__ import annotations

import shutil
from pathlib import Path

from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from smac.acquisition.function import EI
from smac.model.random_forest.random_forest import RandomForest


from .search_space import get_configspace


class BOHBOptimizer:
    """
    SMAC3-backed BOHB optimizer (replaces the flat BayesianOptimizer /
    HyperparameterOptimizationFacade). This is SMAC's "SMAC-HB" facade
    configuration -- SMAC's own implementation of BOHB (Falkner et al.
    2018): Hyperband's successive-halving budget allocation, with BO
    (Random Forest surrogate + Expected Improvement) replacing Hyperband's
    random sampling of configs at each rung.

    External interface:

        optimizer = BOHBOptimizer(
            seed, min_budget=50, max_budget=1000, n_trials=200, eta=3,
            run_id="wine_quality_fold1",   # NEW -- see below
        )
        best_params, best_score = optimizer.optimize(objective_fn)

    where objective_fn(params: dict, budget: float) -> float (higher=better).
    `best_params` never contains a budget/n_estimators key -- that's fixed
    separately afterwards by fit_lgbm_with_early_stopping (see utils.py).

    Design choices (for poster writeup):
        min_budget / max_budget : bounds on n_estimators used as the
                                   fidelity dimension. min_budget should be
                                   low enough for cheap screening but high
                                   enough that R2 still ranks configs
                                   sensibly (worth sanity-checking on a
                                   practice dataset before trusting it on
                                   the exam dataset).
        eta                     : Hyperband's halving rate -- at each rung,
                                   only the top 1/eta fraction of configs
                                   are promoted to the next (eta x larger)
                                   budget. Default 3, matching both the
                                   original Hyperband and BOHB papers.
        incumbent_selection     : "highest_budget" -- the reported best
                                   config must have been evaluated at
                                   max_budget at least once, so the final
                                   reported R2 isn't just a lucky cheap eval.
        walltime_limit          : optional wall-clock cap (seconds) on the
                                   whole search. None (unlimited) by
                                   default. NOTE: per SMAC's own docs,
                                   reproducibility is only guaranteed if
                                   NO wallclock/CPU-time budget is used to
                                   decide when to stop -- so leave this
                                   None whenever you need reproducible
                                   results, and only set it for the actual
                                   timed exam-dataset run (where a small
                                   amount of run-to-run search variance is
                                   an acceptable trade-off for meeting the
                                   24h budget).

    Reproducibility (see run_id below and automl/determinism.py):
        SMAC's docs state plainly that reproducibility requires a single
        worker and no time-based stopping criterion. We enforce the
        former explicitly (n_workers=1) rather than relying on the
        library default, and enforce the latter by leaving
        walltime_limit unset unless the caller opts in. On top of that,
        every optimize() call gets its own isolated output directory
        (keyed by run_id + seed) that is wiped before the run starts, so
        two calls (e.g. two folds, two tasks, or two reruns after a crash)
        can never silently share or resume from another run's SMAC state.
    """

    def __init__(
        self,
        seed: int,
        min_budget: int = 60,
        max_budget: int = 300,
        n_trials: int = 60,
        eta: int = 3,
        n_init: int = 10,
        walltime_limit: float | None = None,
        run_id: str | None = None,
        output_directory: str | Path = "smac3_output",
    ):
        self.seed = seed
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.n_trials = n_trials
        self.eta = eta
        self.n_init = n_init
        self.walltime_limit = walltime_limit
        self.run_id = run_id
        self.output_directory = Path(output_directory)

    def optimize(self, objective_fn):
        cs = get_configspace(seed=self.seed)

        def target_function(config, seed: int = 0, budget: float = None) -> float:
            params = dict(config)
            score = objective_fn(params, budget=budget)
            return -score  # SMAC minimizes -> flip sign of the R^2-based score

        # Give this run a name unique to (run_id, seed) so that separate
        # optimize() calls -- different folds, different tasks, different
        # seeds, or just a rerun after a crash -- never hash to (and
        # therefore never silently resume from / race on) the same
        # smac3_output/<name>/<seed>/ directory. SMAC will auto-load
        # previous data if it finds a matching, already-populated
        # directory for the same name+initial-state; we don't want to
        # depend on that detection logic working perfectly, so we also
        # wipe the directory ourselves before starting.
        run_name = self.run_id if self.run_id is not None else "bohb_run"
        run_dir = self.output_directory / run_name / str(self.seed)
        if run_dir.exists():
            shutil.rmtree(run_dir)

        scenario_kwargs = dict(
            name=run_name,
            output_directory=self.output_directory,
            n_trials=self.n_trials,
            deterministic=True,
            seed=self.seed,
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            # Required for reproducibility -- SMAC docs: "Reproducibility
            # can only be ensured if one worker is used and no time
            # (wallclock or CPU time) is involved." Pinned explicitly
            # rather than relying on the library default.
            n_workers=1,
        )
        if self.walltime_limit is not None:
            scenario_kwargs["walltime_limit"] = self.walltime_limit

        scenario = Scenario(cs, **scenario_kwargs)

        initial_design = MultiFidelityFacade.get_initial_design(
            scenario, n_configs=self.n_init
        )

        acquisition_function = EI()

        model = MultiFidelityFacade.get_model(scenario, n_trees=200)

        intensifier = Hyperband(
            scenario,
            eta=self.eta,
            incumbent_selection="highest_budget",
        )

        smac = MultiFidelityFacade(
            scenario,
            target_function,
            model=model,
            initial_design=initial_design,
            acquisition_function=acquisition_function,
            intensifier=intensifier,
            overwrite=True,
        )

        incumbent = smac.optimize()
        best_loss = smac.validate(incumbent)

        best_params = dict(incumbent)
        best_score = -best_loss

        return best_params, best_score