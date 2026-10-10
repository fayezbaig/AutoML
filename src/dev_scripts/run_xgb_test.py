from automl.phase1_ablations import compare_extended_portfolio
compare_extended_portfolio("xgboost", "bike_sharing_demand", budget=30, seed=0)
