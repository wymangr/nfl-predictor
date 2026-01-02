import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import json

from src.model.features import (
    engineer_features,
    FEATURE_SELECTION_BASE_FEATURES,
    TARGET as target,
)
from src.helpers.database_helpers import run_query


def train_model():
    # ============================================================================
    # LOAD HYPERPARAMETERS FROM TUNING
    # ============================================================================
    print("=" * 70)
    print("LOADING OPTIMIZED HYPERPARAMETERS")
    print("=" * 70)

    # Try to load hyperparameters from tuning
    try:
        with open("best_hyperparameters.json", "r") as f:
            best_hyperparams = json.load(f)
        print("\n✓ Loaded hyperparameters from 'best_hyperparameters.json'")
        print("\nOptimized Hyperparameters:")
        for param, value in sorted(best_hyperparams.items()):
            if isinstance(value, float):
                print(f"  {param:20s}: {value:.4f}")
            else:
                print(f"  {param:20s}: {value}")
        using_optimized = True
    except FileNotFoundError:
        print("\n⚠️  'best_hyperparameters.json' not found")
        print(
            "   Using default hyperparameters (run random_search.py first for optimal results)"
        )
        best_hyperparams = {
            "n_estimators": 278,
            "learning_rate": 0.040150539417696085,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.600501798058987,
            "colsample_bytree": 0.600501798058987,
            "colsample_bylevel": 0.600501798058987,
            "reg_alpha": 0.40200719223594783,
            "reg_lambda": 0.6030107883539217,
            "gamma": 0.10050179805898696,
        }
        using_optimized = False

    # Fixed parameters (same as random_search.py)
    fixed_params = {
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "mae",
        "early_stopping_rounds": 50,
    }

    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    ## OLD SPLIT (OLD DATA)
    # training_data = run_query(f"SELECT * FROM training_data WHERE season < 2025 OR (season == 2025 and week <= 12) ORDER BY season ASC, week ASC")

    ## OLD SPLIT (NEW DATA)
    # training_data = run_query(f"SELECT * FROM training_data WHERE season <= 2025 ORDER BY season ASC, week ASC")

    ## NEW SPLIT
    training_data = run_query(
        "SELECT * FROM training_data WHERE season <= 2025 ORDER BY season ASC, week ASC"
    )

    df = pd.DataFrame(training_data)
    # df["point_differential"] = df["home_score"] - df["away_score"]

    print(f"Loaded {len(df)} games")
    print(f"Seasons: {df['season'].min()} to {df['season'].max()}")
    print(f"Weeks: {df['week'].min()} to {df['week'].max()}")

    # ============================================================================
    # FEATURE ENGINEERING (must match random_search.py)
    # ============================================================================
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    model_data, best_features = engineer_features(df)

    # Use 'All features combined' (same as random_search.py)
    # best_features = features
    print(
        f"Using feature set: 'All features combined' with {len(best_features)} features"
    )

    # Clean data
    model_data = model_data.dropna(subset=best_features + [target])
    print(f"After removing NaN: {len(model_data)} games")

    # Verify temporal ordering
    model_data = model_data.sort_values(["season", "week"]).reset_index(drop=True)

    # ============================================================================
    # TIME-BASED TRAIN/VAL/TEST SPLIT (MUST MATCH random_search.py)
    # ============================================================================
    print("\n" + "=" * 70)
    print("CREATING TIME-BASED TRAIN/VAL/TEST SPLIT")
    print("=" * 70)
    print("⚠️  Using same split as random_search.py: 70% train / 15% val / 15% test")

    model_data_for_split = model_data.copy()

    ## OLD SPLIT
    # train_idx = int(len(model_data_for_split) * 0.7)
    # val_idx = int(len(model_data_for_split) * 0.85)
    # train_df = model_data_for_split.iloc[:train_idx]
    # val_df = model_data_for_split.iloc[train_idx:val_idx]
    # test_df = model_data_for_split.iloc[val_idx:]

    ## NEW SPLIT
    train_df = model_data_for_split[model_data_for_split["season"] <= 2023]  # 2023
    val_df = model_data_for_split[model_data_for_split["season"] == 2024]  # 2024
    test_df = model_data_for_split[model_data_for_split["season"] == 2025]  # 2025

    X_train = train_df[best_features]
    y_train = train_df[target]
    X_val = val_df[best_features]
    y_val = val_df[target]
    X_test = test_df[best_features]
    y_test = test_df[target]

    print(
        f"\nTrain: {len(train_df)} games (Seasons {train_df['season'].min()}-{train_df['season'].max()})"
    )
    print(
        f"Val:   {len(val_df)} games (Seasons {val_df['season'].min()}-{val_df['season'].max()})"
    )
    print(
        f"Test:  {len(test_df)} games (Seasons {test_df['season'].min()}-{test_df['season'].max()})"
    )

    # ============================================================================
    # TRAIN MODEL WITH EARLY STOPPING (same as random_search.py loop)
    # ============================================================================
    print("\n" + "=" * 70)
    print("TRAINING MODEL WITH EARLY STOPPING")
    print("=" * 70)

    # Create model with optimized hyperparameters
    model = XGBRegressor(**best_hyperparams, **fixed_params)

    # Train with early stopping on validation set
    print(f"\nTraining with early stopping (monitoring validation set)...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print(f"✓ Training complete")
    print(f"  Best iteration: {model.best_iteration} / {model.n_estimators}")

    # ============================================================================
    # EVALUATE MODEL (same evaluation as random_search.py)
    # ============================================================================
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Get predictions for all three sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    from src.model.overfitting import regression_metrics

    train_metrics = regression_metrics(y_train, y_train_pred)
    val_metrics = regression_metrics(y_val, y_val_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    print(f"\nTraining Set:")
    print(f"  MAE:  {train_metrics['mae']:.3f} points")
    print(f"  RMSE: {train_metrics['rmse']:.3f} points")
    print(f"  R²:   {train_metrics['r2']:.3f}")

    print(f"\nValidation Set:")
    print(f"  MAE:  {val_metrics['mae']:.3f} points")
    print(f"  RMSE: {val_metrics['rmse']:.3f} points")
    print(f"  R²:   {val_metrics['r2']:.3f}")

    print(f"\nTest Set:")
    print(f"  MAE:  {test_metrics['mae']:.3f} points")
    print(f"  RMSE: {test_metrics['rmse']:.3f} points")
    print(f"  R²:   {test_metrics['r2']:.3f}")

    print(f"\nGeneralization Analysis:")
    print(
        f"  Train → Val gap:  {val_metrics['mae'] - train_metrics['mae']:.3f} "
        f"({((val_metrics['mae'] - train_metrics['mae'])/train_metrics['mae']*100):+.1f}%)"
    )
    print(
        f"  Train → Test gap: {test_metrics['mae'] - train_metrics['mae']:.3f} "
        f"({((test_metrics['mae'] - train_metrics['mae'])/train_metrics['mae']*100):+.1f}%)"
    )
    print(
        f"  Val → Test gap:   {test_metrics['mae'] - val_metrics['mae']:.3f} "
        f"({((test_metrics['mae'] - val_metrics['mae'])/val_metrics['mae']*100):+.1f}%)"
    )

    # ============================================================================
    # OVERFITTING CHECK
    # ============================================================================
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS (Train vs Val)")
    print("=" * 70)

    from src.model.overfitting import check_overfitting

    report = check_overfitting(
        train_mae=train_metrics["mae"],
        val_mae=val_metrics["mae"],
        train_rmse=train_metrics["rmse"],
        val_rmse=val_metrics["rmse"],
        train_r2=train_metrics["r2"],
        val_r2=val_metrics["r2"],
        mae_rel_threshold=0.15,
        mae_abs_threshold=1.5,
        rmse_rel_threshold=0.15,
        rmse_abs_threshold=2.0,
        r2_drop_threshold=0.15,
    )

    # ============================================================================
    # 2025 PREDICTIONS
    # ============================================================================
    print("\n" + "=" * 70)
    print("2025 HOLDOUT EVALUATION")
    print("=" * 70)

    from src.model.predict import get_past_predictions_model

    model_dict = {
        "model": model,
        "features": best_features,
        "base_features": FEATURE_SELECTION_BASE_FEATURES,
    }

    score_2025 = get_past_predictions_model(model_dict)["overall_accuracy"]
    print(f"\n  Overall 2025 Spread Accuracy: {score_2025:.1f}%")

    # ============================================================================
    # PRACTICAL ACCURACY METRICS
    # ============================================================================
    print("\n" + "=" * 70)
    print("PRACTICAL ACCURACY FOR SPREAD BETTING")
    print("=" * 70)

    for set_name, y_true, y_pred in [
        ("Training", y_train, y_train_pred),
        ("Validation", y_val, y_val_pred),
        ("Test", y_test, y_test_pred),
    ]:
        errors = y_true - y_pred
        within_3 = (np.abs(errors) <= 3).mean() * 100
        within_7 = (np.abs(errors) <= 7).mean() * 100
        within_14 = (np.abs(errors) <= 14).mean() * 100

        print(f"\n{set_name} Set:")
        print(f"  Bias (mean error): {errors.mean():.3f}")
        print(f"  Std of errors: {errors.std():.3f}")
        print(f"  Within 3 pts:  {within_3:.1f}%")
        print(f"  Within 7 pts:  {within_7:.1f}%")
        print(f"  Within 14 pts: {within_14:.1f}%")

    # ============================================================================
    # FEATURE IMPORTANCE & NOISE ANALYSIS
    # ============================================================================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE & NOISE ANALYSIS")
    print("=" * 70)

    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": best_features, "importance": importance_scores}
    ).sort_values("importance", ascending=False)

    print("\n📊 TOP 20 MOST IMPORTANT FEATURES:")
    print(importance_df.head(20).to_string(index=False))

    # ============================================================================
    # SAVE MODEL AND METADATA
    # ============================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    val_errors = y_val - y_val_pred

    model_artifacts = {
        "model": model,
        "feature_set_name": "All features combined",
        "features": best_features,
        "base_features": FEATURE_SELECTION_BASE_FEATURES,
        "hyperparameters": best_hyperparams,
        "best_iteration": model.best_iteration,
        "bias": val_errors.mean(),  # Use validation set bias
        "using_optimized_hyperparams": using_optimized,
        "metrics": {
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "train_val_gap": val_metrics["mae"] - train_metrics["mae"],
            "2025_score": score_2025,
            "overfitting": report["overfit"],
            "overfitting_severity": report["severity"],
        },
    }

    with open("nfl-prediction.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)

    print("✓ Saved model to 'nfl-prediction.pkl'")
    print(f"  - Model: XGBoost")
    print(f"  - Feature set: All features combined")
    print(f"  - Features: {len(best_features)}")
    print(f"  - Best iteration: {model.best_iteration}")
    print(f"  - Hyperparameters: {'Optimized' if using_optimized else 'Default'}")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nFeature Set: All features combined")
    print(f"Number of Features: {len(best_features)}")
    print(
        f"Hyperparameters: {'Optimized (from tuning)' if using_optimized else 'Default'}"
    )

    print(f"\nValidation Performance (used for model selection):")
    print(f"  MAE:  {val_metrics['mae']:.3f} points")
    print(f"  RMSE: {val_metrics['rmse']:.3f} points")
    print(f"  R²:   {val_metrics['r2']:.3f}")

    print(f"\nTest Performance (final holdout):")
    print(f"  MAE:  {test_metrics['mae']:.3f} points")
    print(f"  RMSE: {test_metrics['rmse']:.3f} points")
    print(f"  R²:   {test_metrics['r2']:.3f}")

    print(f"\n2025 Holdout:")
    print(f"  Spread Accuracy: {score_2025:.1f}%")

    print(f"\nGeneralization:")
    print(f"  Train MAE: {train_metrics['mae']:.3f}")
    print(f"  Val MAE:   {val_metrics['mae']:.3f}")
    print(f"  Test MAE:  {test_metrics['mae']:.3f}")
    print(
        f"  Train→Val gap: {val_metrics['mae'] - train_metrics['mae']:.3f} "
        f"({((val_metrics['mae'] - train_metrics['mae'])/train_metrics['mae']*100):+.1f}%)"
    )

    if report["overfit"]:
        print(f"\n⚠️  Overfitting detected (Severity: {report['severity']})")
    else:
        print("\n✅ No significant overfitting detected")

    if not using_optimized:
        print(
            "\n💡 TIP: Run 'python random_search.py' first to find optimal hyperparameters"
        )

    print("\n" + "=" * 70)
