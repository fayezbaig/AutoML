"""fANOVA utilities for SMAC/BOHB run histories.

The reported values are *global* hyperparameter importances over evaluated
configurations. For multi-fidelity runs, the analysis prefers observations at
the maximum budget because those are closest to the finally deployed model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _importance_value(raw: Any) -> float:
    """Support both old scalar and newer dict-shaped fanova results."""
    if isinstance(raw, dict):
        for key in ("individual importance", "total importance"):
            if key in raw:
                return float(raw[key])
        for value in raw.values():
            try:
                return _importance_value(value)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"Could not extract fANOVA importance from {raw!r}")
    return float(raw)


def compute_fanova_importance(smac: Any, config_space: Any) -> dict[str, float]:
    """Compute first-order fANOVA importances from a finished SMAC object.

    Failed/non-finite trials are ignored. At least two unique configurations
    are required; in practice 8+ are recommended for a meaningful result.
    """
    try:
        from fanova import fANOVA
    except ImportError as exc:
        raise ImportError(
            "fANOVA analysis requires the optional package 'fanova'. "
            "Install it with: pip install fanova"
        ) from exc

    runhistory = smac.runhistory
    rows: list[tuple[Any, float, float | None]] = []
    for trial_key, trial_value in runhistory.items():
        cost = getattr(trial_value, "cost", None)
        if cost is None or not np.isfinite(float(cost)):
            continue
        config = runhistory.get_config(trial_key.config_id)
        budget = getattr(trial_key, "budget", None)
        rows.append((config, float(cost), None if budget is None else float(budget)))

    if not rows:
        raise ValueError("SMAC run history contains no finite completed trials.")

    budgets = [b for _, _, b in rows if b is not None and np.isfinite(b)]
    if budgets:
        max_budget = max(budgets)
        high_fidelity = [r for r in rows if r[2] is not None and np.isclose(r[2], max_budget)]
        if len(high_fidelity) >= 8:
            rows = high_fidelity

    # A configuration can appear repeatedly due to seeds/budgets. Average its
    # observed cost so the surrogate is not biased by duplicate rows.
    grouped: dict[tuple[float, ...], list[float]] = {}
    for config, cost, _ in rows:
        arr = tuple(np.asarray(config.get_array(), dtype=float).tolist())
        grouped.setdefault(arr, []).append(cost)

    if len(grouped) < 2:
        raise ValueError("Need at least two unique evaluated configurations for fANOVA.")

    X = np.asarray(list(grouped.keys()), dtype=float)
    y = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    analyzer = fANOVA(X, y, config_space=config_space)

    importances: dict[str, float] = {}
    for hp in config_space.values():
        raw = analyzer.quantify_importance((hp.name,))
        importances[hp.name] = max(0.0, _importance_value(raw))

    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def save_importance_histogram(
    importances: dict[str, float],
    selected_config: dict[str, Any],
    output_path: str | Path,
    algorithm: str,
) -> Path:
    """Save a horizontal bar histogram and a JSON sidecar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(importances.keys())
    values = [importances[n] for n in names]
    labels = [f"{n} = {selected_config.get(n, 'fidelity-controlled')}" for n in names]

    fig_height = max(3.2, 0.65 * len(names) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    positions = np.arange(len(names))
    ax.barh(positions, values)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized first-order fANOVA importance")
    ax.set_title(f"Hyperparameter importance — selected {algorithm} configuration")
    ax.set_xlim(0, max(values) * 1.18 if values else 1.0)
    for pos, value in zip(positions, values):
        ax.text(value, pos, f" {value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "algorithm": algorithm,
        "selected_config": selected_config,
        "fanova_importance": importances,
        "note": "Global first-order importance over the SMAC run history; highest-fidelity trials preferred.",
    }, indent=2, default=str), encoding="utf-8")
    return output_path