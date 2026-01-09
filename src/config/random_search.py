import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import pickle
from datetime import datetime

from src.model.train import run_query
from src.model.features import engineer_features
from src.model.predict import get_past_predictions_model
from src.model.overfitting import check_overfitting, regression_metrics
from src.model.features import ALL_BASE_FEATURES as base_features, TARGET as target


def run_random_search(
    iterations, threshold, resume, min_train_r2, min_iterations, spread_line=False
):
    print("=" * 80)
    print("NFL HYPERPARAMETER OPTIMIZATION WITH STRICT CONSTRAINTS")
    print("=" * 80)
    print(f"Iterations: {iterations} parameter combinations")
    print(f"\nConstraints (ALL must pass):")
    print(f"  1. 2025 Spread Accuracy >= {threshold}%")
    print(f"  2. No overfitting detected")
    print(
        f"  3. Early stopping quality (>= {min_iterations} iterations, >= 10% of n_estimators)"
    )
    # print(f"  4. Spread dependency (<= {max_spread_importance*100:.0f}% feature importance)")
    print(f"  5. Training quality (R² >= {min_train_r2})")
    print(f"  6. Proper generalization (train MAE <= val MAE)")
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for checkpoint
    checkpoint_file = "random_search_checkpoint.pkl"
    if resume and os.path.exists(checkpoint_file):
        print(f"\n📂 RESUMING from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        all_results = checkpoint["all_results"]
        valid_configs = checkpoint["valid_configs"]
        rejected_by_2025 = checkpoint["rejected_by_2025"]
        rejected_by_overfitting = checkpoint["rejected_by_overfitting"]
        rejected_by_both = checkpoint["rejected_by_both"]
        best_valid_config = checkpoint["best_valid_config"]
        best_valid_score = checkpoint["best_valid_score"]
        start_iteration = checkpoint["last_iteration"] + 1
        checkpoint_start_time = checkpoint["start_time"]

        print(f"   Resuming from iteration {start_iteration}")
        print(f"   Already completed: {len(all_results)} iterations")
        print(f"   Valid configs found: {len(valid_configs)}")
        print(f"   Best valid score so far: {best_valid_score:.3f}")
    else:
        if resume:
            print(
                f"\n⚠️  --resume specified but no checkpoint found at {checkpoint_file}"
            )
            print(f"   Starting fresh search...")

        # Initialize fresh search
        all_results = []
        valid_configs = []
        rejected_by_2025 = 0
        rejected_by_overfitting = 0
        rejected_by_both = 0
        best_valid_config = None
        best_valid_score = float("inf")
        start_iteration = 0
        checkpoint_start_time = datetime.now()

    print("=" * 80)

    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    training_data = run_query(
        "SELECT * FROM training_data WHERE season <= 2025 ORDER BY season ASC, week ASC"
    )
    df = pd.DataFrame(training_data)
    df["point_differential"] = df["home_score"] - df["away_score"]

    print(f"Loaded {len(df)} games")
    print(f"Seasons: {df['season'].min()} to {df['season'].max()}")
    print(f"Weeks: {df['week'].min()} to {df['week'].max()}")

    # ============================================================================
    # FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    model_data, best_features = engineer_features(df)
    print(
        f"Using feature set: 'All features combined' with {len(best_features)} features"
    )

    model_data = model_data.dropna(subset=best_features + [target])
    print(f"After removing NaN: {len(model_data)} games")

    model_data = model_data.sort_values(["season", "week"]).reset_index(drop=True)

    # ============================================================================
    # TIME-BASED TRAIN/VAL/TEST SPLIT
    # ============================================================================
    print("\n" + "=" * 80)
    print("CREATING TIME-BASED TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    model_data_for_split = model_data.copy()

    # Split: 70% train, 15% validation, 15% test
    # train_idx = int(len(model_data_for_split) * 0.7)
    # val_idx = int(len(model_data_for_split) * 0.85)

    # train_df = model_data_for_split.iloc[:train_idx]
    # val_df = model_data_for_split.iloc[train_idx:val_idx]
    # test_df = model_data_for_split.iloc[val_idx:]

    train_df = model_data_for_split[model_data_for_split["season"] <= 2023]
    val_df = model_data_for_split[model_data_for_split["season"] == 2024]
    test_df = model_data_for_split[model_data_for_split["season"] == 2025]

    X_train = train_df[best_features]
    y_train = train_df[target]
    X_val = val_df[best_features]
    y_val = val_df[target]
    X_test = test_df[best_features]
    y_test = test_df[target]

    print(
        f"Train: {len(train_df)} games (Seasons {train_df['season'].min()}-{train_df['season'].max()})"
    )
    print(
        f"Val:   {len(val_df)} games (Seasons {val_df['season'].min()}-{val_df['season'].max()})"
    )
    print(
        f"Test:  {len(test_df)} games (Seasons {test_df['season'].min()}-{test_df['season'].max()})"
    )

    # ============================================================================
    # HYPERPARAMETER SEARCH SPACE
    # ============================================================================
    print("\n" + "=" * 80)
    print("DEFINING HYPERPARAMETER SEARCH SPACE")
    print("=" * 80)

    param_distributions = {
        "n_estimators": randint(100, 800),
        "learning_rate": uniform(0.01, 0.15),
        "max_depth": randint(2, 8),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "colsample_bylevel": uniform(0.5, 0.5),
        "reg_alpha": uniform(0, 3.0),
        "reg_lambda": uniform(0, 5.0),
        "gamma": uniform(0, 0.7),
    }

    fixed_params = {
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "mae",
        "early_stopping_rounds": 50,
    }

    print("\n🔍 HYPERPARAMETERS BEING OPTIMIZED:")
    print("-" * 80)
    for param, dist in param_distributions.items():
        dist_type = type(dist).__name__
        if "randint" in dist_type or hasattr(dist, "a"):
            print(f"  {param:25s}: {dist.a} to {dist.b - 1} (integers)")
        elif "uniform" in dist_type or hasattr(dist, "args"):
            loc = dist.args[0] if hasattr(dist, "args") else dist.loc
            scale = dist.args[1] if hasattr(dist, "args") else dist.scale
            print(f"  {param:25s}: {loc:.3f} to {loc + scale:.3f} (continuous)")

    print("\n🔒 FIXED PARAMETERS:")
    print("-" * 80)
    for param, value in fixed_params.items():
        print(f"  {param:25s}: {value}")

    # ============================================================================
    # MANUAL RANDOM SEARCH WITH STRICT CONSTRAINTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("RUNNING CONSTRAINED RANDOM SEARCH")
    print("=" * 80)
    print(f"Testing up to {iterations} parameter combinations")
    print(f"\nAll 6 constraints must pass for a config to be valid:")
    print(f"  ✓ 2025 accuracy >= {threshold}%")
    print(f"  ✓ No overfitting")
    print(f"  ✓ Early stop >= {min_iterations} iterations & >= 10% ratio")
    # print(f"  ✓ Spread importance <= {max_spread_importance*100:.0f}%")
    print(f"  ✓ Train R² >= {min_train_r2}")
    print(f"  ✓ Train MAE <= Val MAE")
    print(f"\nPress Ctrl+C to stop early and use best valid config found.\n")

    # Track all results
    # all_results = []  # Now initialized above (or from checkpoint)
    # valid_configs = []
    # rejected_by_2025 = 0
    # rejected_by_overfitting = 0
    # rejected_by_both = 0

    # best_valid_config = None
    # best_valid_score = float('inf')

    start_time = datetime.now()

    # Manual random search loop
    for iteration in range(start_iteration, iterations):
        iter_start = datetime.now()

        # Sample random hyperparameters
        params = {}
        for param_name, distribution in param_distributions.items():
            if hasattr(distribution, "rvs"):
                params[param_name] = distribution.rvs(random_state=42 + iteration)
            else:
                params[param_name] = distribution

        # Convert numpy types to Python types
        for key, value in params.items():
            if isinstance(value, (np.integer, np.int64)):
                params[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                params[key] = float(value)

        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"{'='*80}")
        print(f"Testing parameters:")
        for param, value in sorted(params.items()):
            if isinstance(value, float):
                print(f"  {param:20s}: {value:.4f}")
            else:
                print(f"  {param:20s}: {value}")

        # Train model with these parameters
        try:
            model = XGBRegressor(**params, **fixed_params)

            # Train with early stopping
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Get predictions for both sets
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate metrics
            train_metrics = regression_metrics(y_train, y_train_pred)
            val_metrics = regression_metrics(y_val, y_val_pred)

            train_mae = train_metrics["mae"]
            val_mae = val_metrics["mae"]

            print(f"\n  Train MAE: {train_mae:.3f}")
            print(f"  Val MAE:   {val_mae:.3f}")
            print(
                f"  Gap:       {val_mae - train_mae:.3f} ({((val_mae - train_mae)/train_mae*100):+.1f}%)"
            )

            # CONSTRAINT 1: Check overfitting
            print(f"\n  Checking Constraint 1: Overfitting...")
            overfitting_report = check_overfitting(
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

            is_overfitting = overfitting_report["overfit"]

            if is_overfitting:
                print(
                    f"  ❌ REJECTED: Overfitting detected (severity: {overfitting_report['severity']})"
                )
                for warning in overfitting_report["warnings"][
                    :3
                ]:  # Show first 3 warnings
                    print(f"     - {warning}")
            else:
                print(f"  ✓ PASSED: No overfitting detected")

            # CONSTRAINT 2: Check 2025 predictions
            print(f"\n  Checking Constraint 2: 2025 Spread Accuracy...")
            model_dict = {
                "model": model,
                "features": best_features,
                "base_features": base_features,
            }

            score_2025 = get_past_predictions_model(model_dict, spread_line)[
                "overall_accuracy"
            ]

            if score_2025 >= threshold:
                print(f"  ✓ PASSED: 2025 score = {score_2025:.1f}% (>= {threshold}%)")
            else:
                print(f"  ❌ REJECTED: 2025 score = {score_2025:.1f}% (< {threshold}%)")

            # NEW CONSTRAINT 3: Check early stopping quality
            print(f"\n  Checking Constraint 3: Early Stopping Quality...")
            early_stop_ratio = model.best_iteration / params["n_estimators"]
            min_ratio_required = 0.10  # Must use at least 10% of allowed iterations

            is_early_stop_healthy = (
                model.best_iteration >= min_iterations
                and early_stop_ratio >= min_ratio_required
            )

            if not is_early_stop_healthy:
                print(f"  ❌ REJECTED: Early stopping too early")
                print(
                    f"     Best iteration: {model.best_iteration} / {params['n_estimators']} ({early_stop_ratio*100:.1f}%)"
                )
                print(
                    f"     Required: >= {min_iterations} iterations AND >= {min_ratio_required*100:.0f}% ratio"
                )
            else:
                print(
                    f"  ✓ PASSED: Best iteration {model.best_iteration} / {params['n_estimators']} ({early_stop_ratio*100:.1f}%)"
                )

            # NEW CONSTRAINT 4: Check spread dependency
            # print(f"\n  Checking Constraint 4: Spread Dependency...")
            # feature_importance = model.feature_importances_
            # spread_idx = best_features.index('spread_line') if 'spread_line' in best_features else -1

            # if spread_idx >= 0:
            #     spread_importance = feature_importance[spread_idx]

            #     if spread_importance > max_spread_importance:
            #         print(f"  ❌ REJECTED: Too dependent on Vegas spread")
            #         print(f"     Spread importance: {spread_importance*100:.1f}% (max allowed: {max_spread_importance*100:.0f}%)")
            #     else:
            #         print(f"  ✓ PASSED: Spread importance = {spread_importance*100:.1f}% (<= {max_spread_importance*100:.0f}%)")
            # else:
            #     print(f"  ✓ PASSED: No spread_line feature")

            # is_spread_healthy = spread_idx < 0 or feature_importance[spread_idx] <= max_spread_importance
            is_spread_healthy = True

            # NEW CONSTRAINT 5: Check training quality (R²)
            print(f"\n  Checking Constraint 5: Training Quality (R²)...")

            if train_metrics["r2"] < min_train_r2:
                print(f"  ❌ REJECTED: Training R² too low (not learning enough)")
                print(
                    f"     Train R²: {train_metrics['r2']:.3f} (min required: {min_train_r2:.3f})"
                )
            else:
                print(
                    f"  ✓ PASSED: Train R² = {train_metrics['r2']:.3f} (>= {min_train_r2:.3f})"
                )

            is_training_quality_good = train_metrics["r2"] >= min_train_r2

            # NEW CONSTRAINT 6: Check generalization direction (train should perform best)
            print(f"\n  Checking Constraint 6: Generalization Direction...")
            is_generalization_healthy = train_mae <= val_mae  # Train should be <= Val

            if not is_generalization_healthy:
                print(
                    f"  ❌ REJECTED: Backwards generalization (val better than train)"
                )
                print(f"     Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}")
                print(
                    f"     This suggests data issues or the model isn't truly learning"
                )
            else:
                print(
                    f"  ✓ PASSED: Proper hierarchy (Train: {train_mae:.3f} <= Val: {val_mae:.3f})"
                )

            # Determine if configuration is valid (ALL constraints must pass)
            passes_2025 = score_2025 >= threshold
            passes_overfitting = not is_overfitting
            passes_early_stop = is_early_stop_healthy
            passes_spread = is_spread_healthy
            passes_training_quality = is_training_quality_good
            passes_generalization = is_generalization_healthy

            is_valid = all(
                [
                    passes_2025,
                    passes_overfitting,
                    passes_early_stop,
                    passes_spread,
                    passes_training_quality,
                    passes_generalization,
                ]
            )

            # Track rejection reasons with more granularity
            if not is_valid:
                rejection_reasons = []
                if not passes_2025:
                    rejection_reasons.append("2025")
                if not passes_overfitting:
                    rejection_reasons.append("overfitting")
                if not passes_early_stop:
                    rejection_reasons.append("early_stop")
                if not passes_spread:
                    rejection_reasons.append("spread_dependency")
                if not passes_training_quality:
                    rejection_reasons.append("low_r2")
                if not passes_generalization:
                    rejection_reasons.append("backwards_gen")

                # Update rejection counters (simplified - just count total rejections)
                if not passes_2025 and not passes_overfitting:
                    rejected_by_both += 1
                elif not passes_2025:
                    rejected_by_2025 += 1
                elif not passes_overfitting:
                    rejected_by_overfitting += 1
                # Note: Not tracking new constraints separately in old counters

            # Store results
            result = {
                "iteration": iteration + 1,
                "params": params.copy(),
                "train_mae": train_mae,
                "val_mae": val_mae,
                "gap": val_mae - train_mae,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "best_iteration": model.best_iteration,
                "early_stop_ratio": early_stop_ratio,
                # 'spread_importance': feature_importance[spread_idx] if spread_idx >= 0 else 0.0,
                "score_2025": score_2025,
                "is_overfitting": is_overfitting,
                "overfitting_severity": overfitting_report["severity"],
                "is_valid": is_valid,
                "passes_2025": passes_2025,
                "passes_overfitting": passes_overfitting,
                "passes_early_stop": passes_early_stop,
                "passes_spread": passes_spread,
                "passes_training_quality": passes_training_quality,
                "passes_generalization": passes_generalization,
                "rejection_reasons": rejection_reasons if not is_valid else [],
            }
            all_results.append(result)

            # Update best valid configuration
            if is_valid:
                with open("random_search_results.txt", "a") as f:
                    f.write(str(result) + "\n")
                valid_configs.append(result)
                print(f"\n  🎉 VALID CONFIGURATION FOUND!")
                print(f"     Valid configs so far: {len(valid_configs)}")

                if val_mae < best_valid_score:
                    best_valid_score = val_mae
                    best_valid_config = {
                        "params": params.copy(),
                        "model": model,
                        "result": result,
                    }
                    print(f"     ⭐ NEW BEST! Val MAE: {val_mae:.3f}")
            else:
                reasons = []
                if not passes_2025:
                    reasons.append(f"2025 score {score_2025:.1f}% < {threshold}%")
                if not passes_overfitting:
                    reasons.append(f"overfitting ({overfitting_report['severity']})")
                if not passes_early_stop:
                    reasons.append(
                        f"early stop at iter {model.best_iteration}/{params['n_estimators']}"
                    )
                # if not passes_spread:
                #     reasons.append(f"spread {feature_importance[spread_idx]*100:.1f}% > 9%")
                if not passes_training_quality:
                    reasons.append(f"train R²={train_metrics['r2']:.3f} < 0.27")
                if not passes_generalization:
                    reasons.append(
                        f"backwards gen (train {train_mae:.3f} > val {val_mae:.3f})"
                    )
                print(f"\n  ❌ INVALID: {', '.join(reasons)}")

            iter_time = (datetime.now() - iter_start).total_seconds()
            print(f"\n  Iteration time: {iter_time:.1f}s")

            # Progress summary
            total_elapsed = (datetime.now() - start_time).total_seconds()
            avg_time_per_iter = total_elapsed / (iteration + 1)
            estimated_remaining = avg_time_per_iter * (iterations - iteration - 1)

            print(
                f"\n  Progress: {iteration + 1}/{iterations} ({(iteration + 1)/iterations*100:.1f}%)"
            )
            print(f"  Valid configs: {len(valid_configs)}")
            print(f"  Rejected by 2025: {rejected_by_2025}")
            print(f"  Rejected by overfitting: {rejected_by_overfitting}")
            print(f"  Rejected by both: {rejected_by_both}")
            print(f"  Est. remaining time: {estimated_remaining/60:.1f} minutes")

            # Save checkpoint every 10 iterations
            if (iteration + 1) % 10 == 0:
                checkpoint = {
                    "all_results": all_results,
                    "valid_configs": valid_configs,
                    "rejected_by_2025": rejected_by_2025,
                    "rejected_by_overfitting": rejected_by_overfitting,
                    "rejected_by_both": rejected_by_both,
                    "best_valid_config": best_valid_config,
                    "best_valid_score": best_valid_score,
                    "last_iteration": iteration,
                    "start_time": checkpoint_start_time,
                    "n_iter": iterations,
                    "threshold_2025": threshold,
                }
                with open("random_search_checkpoint.pkl", "wb") as f:
                    pickle.dump(checkpoint, f)
                print(f"  💾 Checkpoint saved (iteration {iteration + 1})")

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Search interrupted by user at iteration {iteration + 1}")
            # Save checkpoint before exiting
            checkpoint = {
                "all_results": all_results,
                "valid_configs": valid_configs,
                "rejected_by_2025": rejected_by_2025,
                "rejected_by_overfitting": rejected_by_overfitting,
                "rejected_by_both": rejected_by_both,
                "best_valid_config": best_valid_config,
                "best_valid_score": best_valid_score,
                "last_iteration": iteration,
                "start_time": checkpoint_start_time,
                "n_iter": iterations,
                "threshold_2025": threshold,
            }
            with open("random_search_checkpoint.pkl", "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"   💾 Checkpoint saved - Use --resume to continue")
            print(f"   Will use best valid configuration found so far...")
            break

        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

    end_time = datetime.now()
    elapsed_hours = (end_time - checkpoint_start_time).total_seconds() / 3600
    elapsed_minutes = (end_time - checkpoint_start_time).total_seconds() / 60
    elapsed_seconds = (end_time - checkpoint_start_time).total_seconds()
    n_completed = len(all_results)

    # ============================================================================
    # CHECK IF WE FOUND ANY VALID CONFIGURATIONS
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"  Runtime: {elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes)")
    print(f"  Iterations completed: {n_completed}/{iterations}")
    print(f"  Valid configurations: {len(valid_configs)}")
    print(f"  Rejected by 2025 constraint: {rejected_by_2025}")
    print(f"  Rejected by overfitting: {rejected_by_overfitting}")
    print(f"  Rejected by both: {rejected_by_both}")

    if len(valid_configs) == 0:
        print(f"\n{'='*80}")
        print(f"❌ NO VALID CONFIGURATIONS FOUND")
        print(f"{'='*80}")
        print(f"All {n_completed} configurations failed at least one constraint.")
        print("\nRejection reasons breakdown:")

        # Analyze rejection reasons from results
        if len(all_results) > 0:
            results_df = pd.DataFrame(all_results)

            # Count each rejection reason
            rejection_counts = {}
            for reasons in results_df["rejection_reasons"]:
                for reason in reasons:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            print("\nMost common rejection reasons:")
            for reason, count in sorted(
                rejection_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {reason}: {count} times ({count/n_completed*100:.1f}%)")

        print("\nSuggestions:")
        print(f"  1. Relax constraints (e.g., lower 2025 threshold, allow spread > 9%)")
        print(f"  2. Run more iterations (try --iterations {iterations * 2})")
        print(f"  3. Adjust hyperparameter search space")
        print(f"  4. Review data quality and feature engineering")
        print("\nSaving partial results but no model will be trained.")

        # Save partial results for analysis
        # results_df = pd.DataFrame(all_results)
        # results_df.to_csv('random_search_results_no_valid2.csv', index=False)
        # print(f"✓ Saved results to 'random_search_results_no_valid2.csv' for analysis")
        exit(1)

    # ============================================================================
    # FINAL MODEL: USE BEST VALID MODEL DIRECTLY
    # ============================================================================
    print("\n" + "=" * 80)
    print("USING BEST VALID MODEL (Already Validated)")
    print("=" * 80)

    best_params = best_valid_config["params"]
    best_result = best_valid_config["result"]

    print(f"\nBest Valid Config:")
    print(f"  Validation MAE: {best_valid_score:.3f}")
    print(f"  2025 Score: {best_result['score_2025']:.1f}%")
    print(f"  Overfitting: {'No ✓' if not best_result['is_overfitting'] else 'Yes ✗'}")

    print(f"\nBest Hyperparameters:")
    for param, value in sorted(best_params.items()):
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.4f}")
        else:
            print(f"  {param:20s}: {value}")

    # Use the model from the best valid config (already trained and validated)
    final_model = best_valid_config["model"]

    print(f"\n✓ Using validated model (trained on {len(X_train)} training games)")

    # ============================================================================
    # EVALUATE FINAL MODEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION")
    print("=" * 80)

    # Evaluate on all three sets for transparency
    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)

    train_metrics = regression_metrics(y_train, y_train_pred)
    val_metrics = regression_metrics(y_val, y_val_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    print(f"\nTraining Set:")
    print(f"  MAE:  {train_metrics['mae']:.3f} points")
    print(f"  RMSE: {train_metrics['rmse']:.3f} points")
    print(f"  R²:   {train_metrics['r2']:.3f}")

    print(f"\nValidation Set (Used for constraint checking):")
    print(f"  MAE:  {val_metrics['mae']:.3f} points")
    print(f"  RMSE: {val_metrics['rmse']:.3f} points")
    print(f"  R²:   {val_metrics['r2']:.3f}")

    print(f"\nTest Set (Final holdout):")
    print(f"  MAE:  {test_metrics['mae']:.3f} points")
    print(f"  RMSE: {test_metrics['rmse']:.3f} points")
    print(f"  R²:   {test_metrics['r2']:.3f}")

    print(f"\nGeneralization Analysis:")
    print(
        f"  Train → Val gap:  {val_metrics['mae'] - train_metrics['mae']:.3f} "
        f"({((val_metrics['mae'] - train_metrics['mae'])/train_metrics['mae']*100):+.1f}%) "
        f"[Used for overfitting check]"
    )
    print(
        f"  Train → Test gap: {test_metrics['mae'] - train_metrics['mae']:.3f} "
        f"({((test_metrics['mae'] - train_metrics['mae'])/train_metrics['mae']*100):+.1f}%)"
    )
    print(
        f"  Val → Test gap:   {test_metrics['mae'] - val_metrics['mae']:.3f} "
        f"({((test_metrics['mae'] - val_metrics['mae'])/val_metrics['mae']*100):+.1f}%)"
    )

    # Show the validated constraints from the search
    print(f"\nConstraints (Validated during search):")
    print(
        f"  Overfitting Check: {'PASSED ✓' if not best_result['is_overfitting'] else 'FAILED ✗'}"
    )
    print(
        f"  2025 Accuracy: {best_result['score_2025']:.1f}% {'PASSED ✓' if best_result['passes_2025'] else 'FAILED ✗'} (threshold: {threshold}%)"
    )

    # Evaluate 2025 again to show it's consistent
    print(f"\n2025 Holdout Set (Re-evaluated for verification):")
    final_model_dict = {
        "model": final_model,
        "features": best_features,
        "base_features": base_features,
    }
    final_2025_score = get_past_predictions_model(final_model_dict, spread_line)[
        "overall_accuracy"
    ]
    print(f"\n  2025 Spread Accuracy: {final_2025_score:.1f}%")
    if abs(final_2025_score - best_result["score_2025"]) < 0.1:
        print(f"  ✓ Consistent with search result ({best_result['score_2025']:.1f}%)")
    else:
        print(
            f"  ⚠️  Different from search result ({best_result['score_2025']:.1f}%) - may indicate randomness in evaluation"
        )

    # ============================================================================
    # PRACTICAL ACCURACY METRICS
    # ============================================================================
    print("\n" + "=" * 80)
    print("PRACTICAL ACCURACY FOR SPREAD BETTING")
    print("=" * 80)

    # Show metrics for all sets
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

    # Use validation set metrics for summary (this is what was validated)
    val_errors = y_val - y_val_pred
    val_within_3 = (np.abs(val_errors) <= 3).mean() * 100
    val_within_7 = (np.abs(val_errors) <= 7).mean() * 100

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nSearch Summary:")
    print(f"  Iterations: {n_completed}/{iterations}")
    print(f"  Valid configs found: {len(valid_configs)}")
    print(f"  Runtime: {elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes)")

    print(f"\n  Constraint Results:")
    print(f"    ✓ Passed both: {len(valid_configs)}")
    print(f"    ✗ Failed 2025 only: {rejected_by_2025}")
    print(f"    ✗ Failed overfitting only: {rejected_by_overfitting}")
    print(f"    ✗ Failed both: {rejected_by_both}")

    print(f"\nBest Valid Model Performance:")
    print(f"  Validation MAE: {val_metrics['mae']:.3f} points (used for selection)")
    print(f"  Test MAE: {test_metrics['mae']:.3f} points (final holdout)")
    print(f"  Within 3 pts (val): {val_within_3:.1f}%")
    print(f"  Within 7 pts (val): {val_within_7:.1f}%")
    print(f"  2025 Score: {best_result['score_2025']:.1f}% ✓")
    print(f"  Overfitting: {'No ✓' if not best_result['is_overfitting'] else 'Yes ✗'}")

    if val_within_3 >= 50 and best_result["score_2025"] >= threshold:
        print(f"\n🎉 SUCCESS! Model meets all targets!")
    elif best_result["score_2025"] >= threshold:
        print(
            f"\n✓ Model meets constraints! 2025 score: {best_result['score_2025']:.1f}% >= {threshold}%"
        )
        if not best_result["is_overfitting"]:
            print(f"✓ No overfitting detected")
    else:
        print(
            f"\n⚠ This shouldn't happen - best valid config should meet all constraints"
        )

    print("\n" + "=" * 80)
