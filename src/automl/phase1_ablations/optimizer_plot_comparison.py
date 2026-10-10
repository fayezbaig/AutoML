"""optimizer_plot_comparison.py"""

from automl.data import Dataset
from automl.core import AutoML  # if you need it
from automl.models import build_model  # match this to wherever you actually put models.py
from automl.search_spaces import SEARCH_SPACES
import numpy as np
import matplotlib.pyplot as plt

rs = np.load("results/rs_curves.npy")  # (n_seeds, budget)
bo = np.load("results/bo_curves.npy")

evals = np.arange(1, rs.shape[1] + 1)

fig, ax = plt.subplots(figsize=(7, 5))
for curves, label, color in [(rs, "Random Search", "tab:orange"), (bo, "NePS BO", "tab:blue")]:
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    ax.plot(evals, mean, label=label, color=color)
    ax.fill_between(evals, mean - std, mean + std, alpha=0.2, color=color)

ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Best validation R² so far")
ax.set_title("Search strategy comparison — yprop_4_1 (RandomForest)")
ax.legend()
fig.tight_layout()
fig.savefig("results/optimizer_comparison.png", dpi=150)
print("Saved results/optimizer_comparison.png")
