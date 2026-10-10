"""
pareto_front.py — identify Pareto-optimal models on (R2, prediction cost).
Uses results ALREADY COLLECTED (no new training). Fill in `candidates`
with your real numbers per dataset before running.
"""
import numpy as np


def is_pareto_optimal(candidates):
    """candidates: list of (name, r2, cost) tuples.
    A point is Pareto-optimal if no other point has >= R2 AND <= cost,
    with at least one strictly better (i.e. nothing dominates it)."""
    optimal = []
    for i, (name_i, r2_i, cost_i) in enumerate(candidates):
        dominated = False
        for j, (name_j, r2_j, cost_j) in enumerate(candidates):
            if i == j:
                continue
            if r2_j >= r2_i and cost_j <= cost_i and (
                    r2_j > r2_i or cost_j < cost_i):
                dominated = True
                break
        if not dominated:
            optimal.append(name_i)
    return optimal


def select_from_pareto(candidates, weight_r2=0.8, weight_cost=0.2):
    """Auto-selection: normalize R2 and cost, pick the Pareto point with
    the best weighted combined score. Used as the AUTOMATED default when
    no human interaction is requested."""
    optimal_names = is_pareto_optimal(candidates)
    optimal_points = [c for c in candidates if c[0] in optimal_names]

    r2_vals = [c[1] for c in optimal_points]
    cost_vals = [c[2] for c in optimal_points]
    r2_min, r2_max = min(r2_vals), max(r2_vals)
    cost_min, cost_max = min(cost_vals), max(cost_vals)

    best_name, best_combined = None, -np.inf
    for name, r2, cost in optimal_points:
        r2_norm = (r2 - r2_min) / (r2_max - r2_min) if r2_max > r2_min else 1.0
        cost_norm = (cost - cost_min) / (cost_max - cost_min) if cost_max > cost_min else 0.0
        combined = weight_r2 * r2_norm - weight_cost * cost_norm
        if combined > best_combined:
            best_combined, best_name = combined, name

    return best_name, optimal_names


if __name__ == "__main__":
    # Real measured values from caruana_ensemble.py, wine_quality, fold=1, seed=0
    # (R2 on validation, predict_time in seconds on 439 test rows)
    # Real measured values: brazilian_houses (test R2, predict cost in seconds)
    candidates = [
        ("LightGBM (SB/VB)", 0.9895, 0.003),
        ("RandomForest (SB/VB)", 0.9900, 0.006),
        ("CatBoost (SB/VB)", 0.9941, 0.0006),
        ("TabPFN", 0.9380, 5.1447),
        ("Caruana Ensemble", 0.9734, 5.1516),
    ]

    optimal_names = is_pareto_optimal(candidates)
    print(f"{'Model':22s} {'R2':>8s} {'Cost(s)':>10s}  Pareto-optimal?")
    for name, r2, cost in candidates:
        flag = "YES" if name in optimal_names else "no (dominated)"
        print(f"{name:22s} {r2:8.4f} {cost:10.2f}  {flag}")

    auto_pick, _ = select_from_pareto(candidates)
    print(f"\nAutomated selection (weighted R2/cost): {auto_pick}")


def estimate_and_check_tabpfn_feasible(
        model, X_sample, n_test_rows, max_predict_seconds=300):
    """
    Measure TabPFN's per-row prediction cost on a SMALL sample, extrapolate
    to the real test set size, and decide whether to include it at all --
    rather than assuming a fixed machine speed. Cutoff (5 min / 300s) is
    generous relative to our 24h total budget (prediction is a one-time
    cost at the very end, not the dominant compute phase), while still
    catching cases where TabPFN's cost would become genuinely impractical.
    """
    import time
    n_probe = min(50, len(X_sample))
    t0 = time.time()
    _ = model.predict(X_sample.iloc[:n_probe])
    probe_time = time.time() - t0

    per_row_cost = probe_time / n_probe
    estimated_full_cost = per_row_cost * n_test_rows

    feasible = estimated_full_cost <= max_predict_seconds
    print(f"TabPFN: measured {per_row_cost*1000:.2f}ms/row on {n_probe} rows -> "
          f"estimated {estimated_full_cost:.1f}s for full {n_test_rows} test rows "
          f"(cutoff={max_predict_seconds}s) -> {'FEASIBLE' if feasible else 'EXCLUDED'}")
    return feasible, estimated_full_cost
