"""
determinism.py — shared reproducibility helpers, applied consistently
across ALL entrypoints (optimizer_comparison.py, ensemble.py,
hyperparam_importance.py, sb_vb_gap.py, catboost_bohb.py, automl.py),
not just the original CASH pipeline. Ports the approach from the
teammate's original determinism.py to cover the full project.
"""
from __future__ import annotations
import os
import sys


def reexec_with_fixed_hash_seed() -> None:
    """Must be called as literally the first line of any entrypoint
    script, before any other import. See original rationale: dict/set
    iteration order depends on PYTHONHASHSEED, which is randomized per
    process by default and can flip near-tied SMAC/ConfigSpace
    tie-breaks between runs."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        import subprocess
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        result = subprocess.run([sys.executable] + sys.argv, env=env)
        sys.exit(result.returncode)


def pin_thread_env() -> None:
    """Call immediately after reexec_with_fixed_hash_seed(), before
    numpy/sklearn/lightgbm/etc. are imported."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
