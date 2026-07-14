# automl/determinism.py
"""
Everything needed to make the BOHB/SMAC search bit-for-bit reproducible
across SEPARATE `python run.py ...` process invocations (not just within
one process). Read top-to-bottom -- each function fixes one layer of the
problem, in the order they need to be applied.

Background: SMAC's own docs are explicit about this --
"Reproducibility can only be ensured if one worker is used and no time
(wallclock or CPU time) is involved" (SMAC3 v2.0.1 docs, Advanced Usage
> Reproducibility). That's necessary but NOT sufficient: even with
n_workers=1 and no walltime_limit, non-determinism can still leak in
from (a) Python's own per-process hash randomization, and (b) any
library that spins up its own thread pool (LightGBM, BLAS/OpenMP-backed
libs used by SMAC's random-forest surrogate).
"""
from __future__ import annotations

import os
import sys
import random


def reexec_with_fixed_hash_seed() -> None:
    """
    PYTHONHASHSEED controls how Python hashes str/bytes/objects, which
    controls the ITERATION ORDER of any dict/set keyed by those hashes.
    By default it's randomized per-process (a security feature), so two
    runs of the identical script with the identical --seed can still
    iterate SMAC/ConfigSpace's internal bookkeeping (run-history lookups,
    "seen configs" sets used by the initial design/intensifier) in a
    DIFFERENT order.

    Where two configs are near-tied on acquisition value, that ordering
    can flip which one gets tried/promoted first -- and because the
    exam's actual config (n_trials=8, n_init=4) barely explores the
    space, one flipped tie-break early on cascades into a completely
    different incumbent by the end of the run.

    random.seed()/np.random.seed() do NOT fix this. PYTHONHASHSEED must
    be set as an environment variable BEFORE the interpreter starts --
    setting os.environ["PYTHONHASHSEED"] mid-process has no effect on
    the already-running process. The only fix from inside a script is to
    check for it and, if unset/wrong, re-launch the same script as a
    fresh subprocess with it pinned. Call this as the very first line of
    your entrypoint, before any other import (numpy, sklearn, smac, ...).

    Uses subprocess.run (NOT os.execv) deliberately: on Windows,
    os.execv builds the new process's command line by naively joining
    sys.argv with spaces and does NOT quote arguments -- so any path
    containing a space (e.g. "C:\\Users\\Jane Doe\\...") gets silently
    split apart and the re-exec fails with a "no such file or directory"
    error pointing at a truncated path. subprocess.run quotes arguments
    correctly on both Windows and POSIX.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        import subprocess

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        result = subprocess.run([sys.executable] + sys.argv, env=env)
        sys.exit(result.returncode)


def pin_thread_env() -> None:
    """
    Belt-and-suspenders alongside utils.LGBM_DETERMINISM_KWARGS (which
    already fixes LightGBM's own histogram-building threads via
    deterministic=True/force_row_wise=True/n_jobs=1). This additionally
    pins any BLAS/OpenMP-backed library -- numpy, scipy, sklearn, and
    SMAC's own random-forest surrogate -- to a single thread, so nothing
    further up the stack silently reintroduces floating-point
    summation-order nondeterminism through its own thread pool.

    Must run before numpy (or anything importing numpy) is imported to
    be fully effective -- call immediately after
    reexec_with_fixed_hash_seed(), still before other imports.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def seed_everything(seed: int) -> None:
    """Global-RNG safety net (data loading / anything not given an
    explicit random_state). NOT a substitute for the two functions
    above -- this only handles Python/numpy's *own* RNG state, not hash
    randomization or thread-level nondeterminism."""
    random.seed(seed)
    import numpy as np  # local import: numpy must not be imported before
                        # pin_thread_env() has had a chance to run
    np.random.seed(seed)