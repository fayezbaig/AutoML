"""
search_spaces.py — SINGLE SOURCE OF TRUTH for hyperparameter bounds
across every algorithm and every search method used in this project.

Previously, bounds were duplicated across optimizer_comparison.py,
extended_portfolio.py, catboost_bohb.py, and joint_cash.py -- a real
risk of silent inconsistency (e.g. changing LightGBM's learning_rate
bound in one file but not another). Centralizing here also directly
prevents the kind of stale/duplicate search-space file flagged in our
external review (the old search_space.py prototype).

Two representations are provided per algorithm:
  - RANDOM_BOUNDS: simple (kind, lo, hi) tuples, used by our own
    lightweight random-search sampler (sample_random_config).
  - build_configspace(algo_name): a real ConfigSpace object, used by
    SMAC. exclude_fidelity=True omits the round-count hyperparameter
    (n_estimators/iterations) since that's controlled externally as
    the fidelity dimension in multi-fidelity searches.
"""
import numpy as np


RANDOM_BOUNDS = {
    "lightgbm": {
        "learning_rate": ("log_float", 1e-3, 0.3),
        "num_leaves": ("int", 8, 256),
        "max_depth": ("int", 3, 12),
        "min_child_samples": ("int", 5, 100),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 2, 32),
        "min_samples_leaf": ("int", 1, 20),
    },
    "catboost": {
        "iterations": ("int", 50, 300),
        "depth": ("int", 3, 10),
        "learning_rate": ("log_float", 1e-3, 0.3),
        "l2_leaf_reg": ("log_float", 1e-2, 10.0),
    },
    "xgboost": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 1e-3, 0.3),
        "min_child_weight": ("int", 1, 20),
    },
    "elasticnet": {
        "alpha": ("log_float", 1e-4, 10.0),
        "l1_ratio": ("float", 0.0, 1.0),
    },
}


def sample_random_config(bounds, rng):
    """Shared random-config sampler, used by every random-search baseline
    across the project (previously duplicated in optimizer_comparison.py
    AND extended_portfolio.py)."""
    config = {}
    for name, spec in bounds.items():
        kind, lo, hi = spec
        if kind == "int":
            config[name] = int(rng.integers(lo, hi + 1))
        elif kind == "float":
            config[name] = float(rng.uniform(lo, hi))
        elif kind == "log_float":
            config[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return config


def build_configspace(algo_name, seed=0, exclude_fidelity=False):
    """Real ConfigSpace object for SMAC. exclude_fidelity=True omits the
    algorithm's round-count hyperparameter (n_estimators / iterations),
    since that's supplied externally as the multi-fidelity budget.

    NOTE: ranges widened from initial defaults (WIDENED comments mark
    each change) after investigation showed the original ranges were
    not the bottleneck on our lower-performing datasets -- widened
    regardless, to rule this out more thoroughly and give the search
    genuinely more room to explore."""
    from ConfigSpace import ConfigurationSpace, Integer, Float
    cs = ConfigurationSpace(seed=seed)

    if algo_name == "lightgbm":
        cs.add([
            # WIDENED from (1e-3, 0.3)
            Float("learning_rate", (1e-4, 0.5), log=True),
            # WIDENED from (8, 256)
            Integer("num_leaves", (4, 512)),
            # WIDENED from (3, 12)
            Integer("max_depth", (2, 20)),
            # WIDENED from (5, 100)
            Integer("min_child_samples", (2, 200)),
        ])
    elif algo_name == "random_forest":
        params = [Integer("max_depth", (2, 50)),                      # WIDENED from (2, 32)
                  Integer("min_samples_leaf", (1, 40))]                # WIDENED from (1, 20)
        if not exclude_fidelity:
            params.insert(0, Integer("n_estimators", (50, 500))
                          )       # WIDENED from (50, 300)
        cs.add(params)
    elif algo_name == "catboost":
        cs.add([
            # WIDENED from (1e-3, 0.3)
            Float("learning_rate", (1e-4, 0.5), log=True),
            # WIDENED from (3, 10)
            Integer("depth", (2, 12)),
            # WIDENED from (1e-2, 10.0)
            Float("l2_leaf_reg", (1e-3, 30.0), log=True),
        ])
    elif algo_name == "xgboost":
        params = [
            Float("learning_rate", (1e-4, 0.5), log=True),
            Integer("max_depth", (2, 20)),
            Integer("min_child_weight", (1, 40)),
        ]
        if not exclude_fidelity:
            params.append(Integer("n_estimators", (50, 500)))
        cs.add(params)
    else:
        raise ValueError(f"No ConfigSpace defined for algorithm: {algo_name}")

    return cs


def build_joint_configspace(seed=0):
    """The joint CASH ConfigurationSpace: algorithm choice as a top-level
    categorical, with per-algorithm hyperparameters conditional on it.
    Used by joint_cash.py's single shared SMAC-BOHB search."""
    from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float, EqualsCondition

    cs = ConfigurationSpace(seed=seed)
    algorithm = Categorical(
        "algorithm", [
            "lightgbm", "random_forest", "catboost"])
    cs.add([algorithm])

    lgb_lr = Float("lgb_learning_rate", (1e-3, 0.3), log=True)
    lgb_leaves = Integer("lgb_num_leaves", (8, 256))
    lgb_depth = Integer("lgb_max_depth", (3, 12))
    lgb_min_child = Integer("lgb_min_child_samples", (5, 100))
    cs.add([lgb_lr, lgb_leaves, lgb_depth, lgb_min_child])
    cs.add([
        EqualsCondition(lgb_lr, algorithm, "lightgbm"),
        EqualsCondition(lgb_leaves, algorithm, "lightgbm"),
        EqualsCondition(lgb_depth, algorithm, "lightgbm"),
        EqualsCondition(lgb_min_child, algorithm, "lightgbm"),
    ])

    rf_depth = Integer("rf_max_depth", (2, 32))
    rf_min_leaf = Integer("rf_min_samples_leaf", (1, 20))
    cs.add([rf_depth, rf_min_leaf])
    cs.add([
        EqualsCondition(rf_depth, algorithm, "random_forest"),
        EqualsCondition(rf_min_leaf, algorithm, "random_forest"),
    ])

    cb_lr = Float("cb_learning_rate", (1e-3, 0.3), log=True)
    cb_depth = Integer("cb_depth", (3, 10))
    cb_l2 = Float("cb_l2_leaf_reg", (1e-2, 10.0), log=True)
    cs.add([cb_lr, cb_depth, cb_l2])
    cs.add([
        EqualsCondition(cb_lr, algorithm, "catboost"),
        EqualsCondition(cb_depth, algorithm, "catboost"),
        EqualsCondition(cb_l2, algorithm, "catboost"),
    ])

    return cs
