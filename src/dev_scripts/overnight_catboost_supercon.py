"""Overnight job: proper CatBoost search for superconductivity, with a
real walltime cap this time (previous 2 attempts had to be manually
killed after projecting ~12h / running 2+ hours with no bound)."""
import time
from automl.data import Dataset
from automl.search_methods import run_catboost_bohb, DATADIR
from automl.preprocessing import preprocess_for_algo
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import numpy as np

# 3 hours per fold, 10 folds -- generous but BOUNDED, unlike previous attempts
WALLTIME_PER_FOLD = 30 * 60  # 30 min/fold x 10 folds = max 5h total, fits your ~6h window

test_scores = []
for fold in range(4, 11):  # folds 1-3 already completed and saved separately
    print(f"\n=== Fold {fold}/10 ===", flush=True)
    ds = Dataset.load(datadir=DATADIR, task="superconductivity", fold=fold)
    X_train, X_test, cat_features = preprocess_for_algo("catboost", ds.X_train, ds.X_test)
    y_train, y_test = ds.y_train, ds.y_test

    t0 = time.time()
    best_config = run_catboost_bohb(
        X_train, y_train, cat_features, seed=0,
        root_dir=f"results/overnight_catboost_supercon_fold{fold}",
        walltime_limit=WALLTIME_PER_FOLD,
    )
    elapsed = time.time() - t0
    print(f"Fold {fold} search took {elapsed:.0f}s, config={best_config}", flush=True)

    model = CatBoostRegressor(random_state=0, verbose=False, thread_count=1,
                              cat_features=cat_features, **best_config)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    test_scores.append(score)
    print(f"Fold {fold} TEST R2: {score:.4f}", flush=True)

    # Save progress after EVERY fold, not just at the end -- protects
    # against exactly the kind of interruption that happened tonight
    # (system sleep mid-run losing all progress).
    import numpy as np
    np.save("results/catboost_superconductivity_smac_bohb_test_b30_seed0_partial.npy",
            np.array(test_scores))

test_scores = np.array(test_scores)
print(f"\n=== FINAL: catboost/superconductivity TEST R2: "
      f"{test_scores.mean():.4f} ± {test_scores.std():.4f} ===")

np.save("results/catboost_superconductivity_smac_bohb_test_b30_seed0.npy", test_scores)
print("Saved to results/catboost_superconductivity_smac_bohb_test_b30_seed0.npy")
