"""
SIMPLE FEATURE SELECTION - NO BULLSHIT

This script:
1. Loads data EXACTLY like train.py
2. Tests different feature subsets
3. Reports REAL 2025 accuracy by training and predicting
4. Spits out the best features for you to copy into train script

NO IMPORTS FROM TRAIN SCRIPT. PURE COPY-PASTE.
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import json
from datetime import datetime

from src.model.features import engineer_features
from src.model.predict import get_past_predictions_model
from src.model.features import ALL_BASE_FEATURES as base_features, TARGET as target
from src.helpers.database_helpers import run_query

# ============================================================================
# EXACT COPY FROM TRAIN SCRIPT
# ============================================================================


def run_feature_selection():

    # Load hyperparameters
    with open("best_hyperparameters.json", "r") as f:
        params = json.load(f)

    fixed_params = {
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "mae",
        "early_stopping_rounds": 50,
    }

    # Load data
    print("Loading data...")
    training_data = run_query(
        "SELECT * FROM training_data WHERE season <= 2025 ORDER BY season ASC, week ASC"
    )
    df = pd.DataFrame(training_data)
    df["point_differential"] = df["home_score"] - df["away_score"]
    print(f"Loaded {len(df)} games")

    # Engineer features
    print("\nEngineering features...")
    model_data, all_features = engineer_features(df)
    print(f"Got {len(all_features)} features after engineering")

    # Clean and split data
    model_data = model_data.dropna(subset=all_features + [target])
    model_data = model_data.sort_values(["season", "week"]).reset_index(drop=True)

    train_df = model_data[model_data["season"] <= 2023]
    val_df = model_data[model_data["season"] == 2024]
    test_df = model_data[model_data["season"] == 2025]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Val:   {len(val_df)} games")
    print(f"Test:  {len(test_df)} games")

    # ============================================================================
    # FUNCTION TO TEST FEATURE SUBSET
    # ============================================================================

    def test_features(features, name="Test"):
        """Train model with features and return TRUE 2025 accuracy"""
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"Features: {len(features)}")
        print("=" * 70)

        X_train = train_df[features]
        y_train = train_df[target]
        X_val = val_df[features]
        y_val = val_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # Train model
        model = XGBRegressor(**params, **fixed_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Get metrics
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))

        print(f"Train MAE: {train_mae:.3f}")
        print(f"Val MAE:   {val_mae:.3f}")
        print(f"Test MAE:  {test_mae:.3f}")

        # Get REAL 2025 accuracy
        model_dict = {
            "model": model,
            "features": features,
            "base_features": base_features,
        }

        score_2025 = get_past_predictions_model(model_dict)["overall_accuracy"]
        print(f"\n*** 2025 Accuracy: {score_2025:.2f}% ***")

        return {
            "name": name,
            "features": features,
            "n_features": len(features),
            "train_mae": train_mae,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "score_2025": score_2025,
            "model": model,
        }

    # ============================================================================
    # FEATURE SELECTION METHODS
    # ============================================================================

    print("\n" + "=" * 70)
    print("SIMPLE FEATURE SELECTION")
    print("=" * 70)

    results = []

    # Baseline: All features
    result = test_features(all_features, "Baseline (All 53 features)")
    results.append(result)

    # Method 1: Remove noise features (those with negative permutation importance)
    print("\n\nMethod 1: Remove High Priority Noise Features")
    print("-" * 70)
    print("Training model to get permutation importance...")

    X_train_all = train_df[all_features]
    y_train_all = train_df[target]
    X_val_all = val_df[all_features]
    y_val_all = val_df[target]

    model_all = XGBRegressor(**params, **fixed_params)
    model_all.fit(
        X_train_all, y_train_all, eval_set=[(X_val_all, y_val_all)], verbose=False
    )

    perm_importance = permutation_importance(
        model_all,
        X_val_all,
        y_val_all,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    # Find features with negative permutation importance (actively harmful)
    perm_df = pd.DataFrame(
        {"feature": all_features, "perm_importance": perm_importance.importances_mean}
    ).sort_values("perm_importance")

    print("\nPermutation Importance (bottom 15):")
    print(perm_df.head(15))

    # Remove features with perm_importance < -0.010 (HIGH priority noise)
    high_priority_noise = perm_df[perm_df["perm_importance"] < -0.010][
        "feature"
    ].tolist()
    print(f"\nRemoving {len(high_priority_noise)} HIGH priority noise features:")
    for f in high_priority_noise:
        print(f"  - {f}")

    features_no_high_noise = [f for f in all_features if f not in high_priority_noise]
    result = test_features(
        features_no_high_noise,
        f"No High Noise ({len(features_no_high_noise)} features)",
    )
    results.append(result)

    # Method 2: Remove ALL negative permutation importance features
    all_noise = perm_df[perm_df["perm_importance"] < 0]["feature"].tolist()
    print(f"\n\nMethod 2: Remove ALL {len(all_noise)} negative permutation features:")
    for f in all_noise:
        perm_val = perm_df[perm_df["feature"] == f]["perm_importance"].values[0]
        print(f"  - {f} ({perm_val:.6f})")

    features_no_noise = [f for f in all_features if f not in all_noise]
    result = test_features(
        features_no_noise, f"No Negative Perm ({len(features_no_noise)} features)"
    )
    results.append(result)

    # Method 3: Keep only top N by permutation importance
    for n in [30, 25, 20, 15]:
        top_features = perm_df.nlargest(n, "perm_importance")["feature"].tolist()
        result = test_features(top_features, f"Top {n} by Perm Importance")
        results.append(result)

    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================

    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(
        [
            {
                "Method": r["name"],
                "Features": r["n_features"],
                "Val MAE": r["val_mae"],
                "Test MAE": r["test_mae"],
                "2025 Accuracy": r["score_2025"],
            }
            for r in results
        ]
    )

    print("\n" + results_df.to_string(index=False))

    # Find best by 2025 accuracy
    best = max(results, key=lambda x: x["score_2025"])

    print("\n" + "=" * 70)
    print(f"BEST: {best['name']}")
    print("=" * 70)
    print(f"Features: {best['n_features']}")
    print(f"Val MAE: {best['val_mae']:.3f}")
    print(f"Test MAE: {best['test_mae']:.3f}")
    print(f"2025 Accuracy: {best['score_2025']:.2f}%")

    print("\n\nCOPY THESE FEATURES TO train.py:")
    print("=" * 70)
    print("\n# Comment out the current base_features = [20 features]")
    print("# Replace with:")
    print("\nbase_features = [")
    for f in best["features"]:
        print(f"    '{f}',")
    print("]")

    print("\n\nOr if you want to use it as best_features AFTER engineering:")
    print("=" * 70)
    print("\n# In train.py, after engineer_features():")
    print("# model_data, feature_sets = engineer_features(df, base_features)")
    print("# Instead of: best_features = feature_sets['All features combined']")
    print("# Use:")
    print("\nbest_features = [")
    for f in best["features"]:
        print(f"    '{f}',")
    print("]")

    # Save to file
    output_file = "optimal_features_simple.txt"
    with open(output_file, "w") as f:
        f.write(f"SIMPLE FEATURE SELECTION RESULTS\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 70 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"BEST: {best['name']}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Features: {best['n_features']}\n")
        f.write(f"2025 Accuracy: {best['score_2025']:.2f}%\n\n")
        f.write("base_features = [\n")
        for feat in best["features"]:
            f.write(f"    '{feat}',\n")
        f.write("]\n")

    print(f"\n\nResults saved to: {output_file}")
    print("=" * 70)
