"""
sb_vb_gap.py — Single Best (SB) vs Virtual Best (VB) analysis, as taught
in Assignment 6 (Algorithm Selection).

SB: pick ONE algorithm, fixed across ALL datasets, whichever has the best
    AVERAGE performance overall.
VB: an oracle that picks the best-performing algorithm PER DATASET.
The SB-VB gap quantifies how much value per-dataset algorithm selection
could theoretically add over just always using one fixed algorithm.

Reads your existing saved .npy results -- no new training needed.
"""
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

ALGO_RESULT_FILES = {
    "lightgbm": "lightgbm_{task}_smac_bohb_test_b30_seed0.npy",
    "random_forest": "random_forest_{task}_smac_bo_test_b30_seed0.npy",
    "catboost": "catboost_{task}_smac_bohb_test_b30_seed0.npy",
}

TASKS = ["yprop_4_1", "wine_quality", "bike_sharing_demand",
         "brazilian_houses", "superconductivity"]


# CatBoost's superconductivity BOHB search was aborted (projected ~12h);
# a random-search-only fallback was used instead for this one cell.
# Noted explicitly since it's not directly comparable to the BOHB-tuned
# entries elsewhere -- included anyway for SB/VB completeness.
SPECIAL_CASE_FILES = {
    ("catboost", "superconductivity"): "catboost_superconductivity_rs_test_b30_seed0.npy",
}


def load_mean_score(task, algo, filename_template):
    if (algo, task) in SPECIAL_CASE_FILES:
        fname = SPECIAL_CASE_FILES[(algo, task)]
    else:
        fname = filename_template.format(task=task)
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    scores = np.load(path)
    return float(scores.mean())


def build_score_table(tasks, algo_files):
    table = {}
    for task in tasks:
        table[task] = {}
        for algo, template in algo_files.items():
            score = load_mean_score(task, algo, template)
            if score is not None:
                table[task][algo] = score
    return table


def compute_sb_vb(table):
    all_algos = set()
    for task_scores in table.values():
        all_algos.update(task_scores.keys())
    all_algos = sorted(all_algos)

    algo_averages = {}
    for algo in all_algos:
        scores = [table[task][algo] for task in table if algo in table[task]]
        algo_averages[algo] = np.mean(scores) if scores else -np.inf

    sb_algo = max(algo_averages, key=algo_averages.get)
    sb_score = algo_averages[sb_algo]

    vb_scores = []
    vb_choices = {}
    for task, algo_scores in table.items():
        if not algo_scores:
            continue
        best_algo = max(algo_scores, key=algo_scores.get)
        vb_choices[task] = (best_algo, algo_scores[best_algo])
        vb_scores.append(algo_scores[best_algo])
    vb_score = np.mean(vb_scores)

    return sb_algo, sb_score, vb_score, vb_choices, algo_averages


if __name__ == "__main__":
    table = build_score_table(TASKS, ALGO_RESULT_FILES)

    print("=== Per-dataset scores ===")
    for task, scores in table.items():
        formatted = ", ".join(f"{a}={s:.4f}" for a, s in scores.items())
        print(f"{task:22s} {formatted}")

    sb_algo, sb_score, vb_score, vb_choices, algo_averages = compute_sb_vb(table)

    print("\n=== Algorithm averages (across available datasets) ===")
    for algo, avg in sorted(algo_averages.items(), key=lambda x: -x[1]):
        print(f"{algo:15s} avg R2 = {avg:.4f}")

    print(f"\n=== Single Best (SB) ===")
    print(f"SB algorithm: {sb_algo}")
    print(f"SB average R2: {sb_score:.4f}")

    print(f"\n=== Virtual Best (VB) ===")
    for task, (algo, score) in vb_choices.items():
        print(f"  {task:22s} best={algo:15s} R2={score:.4f}")
    print(f"VB average R2: {vb_score:.4f}")

    gap = vb_score - sb_score
    print(f"\n=== SB-VB Gap ===")
    print(f"Gap = {gap:.4f}")
    print(f"This is the theoretical upper bound on how much per-dataset")
    print(f"algorithm selection could improve over always using {sb_algo}.")
