"""hpo_convergence.py — anytime performance plot: best internal-CV R^2
found so far vs. number of configurations evaluated, comparing Random
Search against SMAC-BOHB, on ONE representative algorithm/dataset/fold.

Kept to a single fold deliberately -- this is an illustrative
methodology plot backing your Motivation section's claim that "random
search wastes effort on poor configurations," not a full benchmark
run. Pick whichever (algo, task) combo shows the clearest gap; wine_quality
is a reasonable default since your other results show a large pipeline
gain there.

WHY THIS NEEDED WRITING (not just reading existing output):
  - SMAC's Scenario(..., output_directory=root_dir) ALREADY writes a
    full run history to disk for every SMAC mode in optimizer_comparison.py
    -- run_smac_bohb_with_trajectory below extracts it directly from
    smac.runhistory after optimize() completes (evaluation order,
    running-min of cost = running-max of R^2), rather than re-deriving
    anything.
  - run_random_search in optimizer_comparison.py only returns the FINAL
    best config -- it never kept the trajectory. run_random_search_with_trajectory
    below is a copy of that same loop with one line added: record the
    best-score-so-far after every trial.

Usage:
    python hpo_convergence.py --algo lightgbm --task wine_quality --fold 1
    python hpo_convergence.py --algo lightgbm --task wine_quality --fold 1 --rs-budget 60 --bohb-trials 60
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")
from automl.determinism import reexec_with_fixed_hash_seed, pin_thread_env
reexec_with_fixed_hash_seed()
pin_thread_env()

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from automl.data import Dataset
from automl.preprocessing import encode_categoricals
from automl.optimizer_comparison import (
    ALGORITHMS,
    sample_random_config,
    internal_cv_score,
    _build_configspace,
    DATADIR,
)


def run_random_search_with_trajectory(algo_name, X_train, y_train, seed, budget):
    """Identical search to optimizer_comparison.run_random_search, but
    also records the best-score-found-so-far after every trial."""
    rng = np.random.default_rng(seed)
    bounds = ALGORITHMS[algo_name]["random_bounds"]
    best_score = -np.inf
    trajectory = []
    for _ in range(budget):
        config = sample_random_config(bounds, rng)
        score = internal_cv_score(algo_name, config, X_train, y_train, seed)
        if score > best_score:
            best_score = score
        trajectory.append(best_score)
    return trajectory


def run_smac_bohb_with_trajectory(
        algo_name, X_train, y_train, seed, root_dir,
        min_budget=60, max_budget=300, n_trials=60, eta=3, n_init=10):
    """Identical search to optimizer_comparison.run_smac_bohb, but also
    extracts the best-score-found-so-far trajectory (in evaluation
    order) from SMAC's own RunHistory after optimize() completes."""
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband
    from smac.acquisition.function import EI

    cs = _build_configspace(algo_name, seed, exclude_n_estimators=True)

    def train(config, seed: int = 0, budget: float = max_budget):
        return -internal_cv_score(
            algo_name, dict(config), X_train, y_train, seed, budget=budget)

    scenario = Scenario(
        cs, name="bohb_convergence", output_directory=root_dir,
        n_trials=n_trials, deterministic=True, seed=seed,
        min_budget=min_budget, max_budget=max_budget, n_workers=1,
    )
    initial_design = MultiFidelityFacade.get_initial_design(
        scenario, n_configs=n_init)
    model = MultiFidelityFacade.get_model(scenario, n_trees=200)
    intensifier = Hyperband(
        scenario, eta=eta, incumbent_selection="highest_budget")

    smac = MultiFidelityFacade(
        scenario, train, model=model, initial_design=initial_design,
        acquisition_function=EI(), intensifier=intensifier, overwrite=True,
    )
    smac.optimize()

    # RunHistory cost = -R^2 (we minimized the negative), so running MIN
    # of cost, in evaluation order, = running MAX of R^2 = the "best
    # score found so far" trajectory.
    entries = list(smac.runhistory.items())
    best_cost = np.inf
    trajectory = []
    for _, trial_value in entries:
        if trial_value.cost < best_cost:
            best_cost = trial_value.cost
        trajectory.append(-best_cost)
    return trajectory


def main(algo, task, fold, seed, rs_budget, bohb_trials):
    ds = Dataset.load(datadir=DATADIR, task=task, fold=fold)
    X_train = encode_categoricals(ds.X_train)
    y_train = ds.y_train

    print(f"[{algo}/{task} fold {fold}] running random search "
          f"({rs_budget} trials)...")
    rs_traj = run_random_search_with_trajectory(
        algo, X_train, y_train, seed, rs_budget)

    print(f"[{algo}/{task} fold {fold}] running SMAC-BOHB "
          f"({bohb_trials} trials)...")
    root_dir = f"results/hpo_convergence_{algo}_{task}_fold{fold}"
    bohb_traj = run_smac_bohb_with_trajectory(
        algo, X_train, y_train, seed, root_dir, n_trials=bohb_trials)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(rs_traj) + 1), rs_traj, label="Random search",
            color="#eb6834", linewidth=2)
    ax.plot(range(1, len(bohb_traj) + 1), bohb_traj, label="SMAC-BOHB",
            color="#1baf7a", linewidth=2)
    ax.set_xlabel("Configurations evaluated")
    ax.set_ylabel("Best internal CV R\u00b2 found so far")
    ax.set_title(f"HPO convergence \u2014 {algo} / {task} (fold {fold})",
                 fontsize=13, loc="left")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    Path("results").mkdir(exist_ok=True)
    out_path = f"results/hpo_convergence_{algo}_{task}_fold{fold}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    print(f"Saved plot to {out_path}")

    # Raw trajectories too, so you can re-plot or combine with other
    # datasets later without rerunning the search.
    np.save(
        f"results/hpo_convergence_{algo}_{task}_fold{fold}_rs.npy",
        np.array(rs_traj))
    np.save(
        f"results/hpo_convergence_{algo}_{task}_fold{fold}_bohb.npy",
        np.array(bohb_traj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=list(ALGORITHMS.keys()), default="lightgbm")
    parser.add_argument("--task", default="wine_quality")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rs-budget", type=int, default=60,
        help="Random search trial count. Match --bohb-trials for a "
             "directly comparable x-axis.")
    parser.add_argument("--bohb-trials", type=int, default=60)
    args = parser.parse_args()

    main(args.algo, args.task, args.fold, args.seed,
         args.rs_budget, args.bohb_trials)