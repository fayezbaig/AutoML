# search_space.py
from __future__ import annotations

from ConfigSpace import ConfigurationSpace, Float, Integer


def get_configspace(seed: int = 42) -> ConfigurationSpace:
    """
    Same bounds as the original manual SearchSpace.sample():
        learning_rate     ~ Uniform(0.01, 0.20)
        num_leaves        ~ Randint(20, 100)
        max_depth         ~ Randint(3, 15)
        min_child_samples ~ Randint(10, 100)

    SMAC/ConfigSpace samples from this directly -- no more manual
    to_array()/from_array() bookkeeping needed, SMAC handles that
    internally.
    """
    cs = ConfigurationSpace(seed=seed)
    cs.add(
        [
            Float("learning_rate", (0.01, 0.20), default=0.1),
            Integer("num_leaves", (20, 100), default=31),
            Integer("max_depth", (3, 15), default=8),
            Integer("min_child_samples", (10, 100), default=20),
        ]
    )
    return cs