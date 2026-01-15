"""
Optimize uncertainty formula to maximize confidence points for correct predictions.

This module analyzes the past_predictions table and searches for the optimal
uncertainty formula by testing thousands of weight and scale combinations.
The goal is to maximize: SUM(confidence) WHERE correct = '1'
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from time import time
import signal
from src.helpers.database_helpers import run_query
from src.model.predict import get_confidence_score


def worker_init():
    """Initialize worker process to ignore Ctrl+C - let main process handle it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_current_confidence_total():
    """Get the current total confidence points for correct predictions."""
    result = run_query(
        "SELECT SUM(confidence) as total_confidence FROM past_predictions WHERE correct = '1'"
    )
    return result[0]["total_confidence"]


def get_available_features(top_n=None, required_features=None):
    """
    Discover all available numeric columns from past_predictions table.
    Optionally filter to top N features by correlation with correct predictions.

    Args:
        top_n: If set, return only top N features by correlation strength
        required_features: Features that should always be included

    Returns:
        list of column names that can be used as features
    """
    # Get a sample row to inspect column types
    sample = run_query("SELECT * FROM past_predictions LIMIT 1")
    if not sample:
        raise ValueError("No data in past_predictions table")

    all_columns = list(sample[0].keys())

    # Columns to exclude from feature selection
    exclude_columns = {
        "game_id",
        "season",
        "week",
        "correct",
        "confidence",
        "confidence_score",
        "home_team",
        "away_team",
        "predicted_winner",
        "spread_favorite",
        "div_game",
        "home_qb_changed",
        "away_qb_changed",
        "spread_diff",
    }

    # Get all data to check correlation
    all_data = run_query("SELECT * FROM past_predictions WHERE correct != 'push'")
    df_all = pd.DataFrame(all_data)

    # Convert correct to numeric
    df_all["correct"] = df_all["correct"].apply(
        lambda x: 1 if x == "1" or x == 1 or x == True else 0
    )

    # Find numeric columns and calculate correlations
    feature_correlations = []
    for col in all_columns:
        if col not in exclude_columns:
            try:
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df_all[col], errors="coerce")
                # Check if column has any non-null numeric values
                if numeric_col.notna().any():
                    # Calculate correlation with "correct" predictions
                    # Use absolute value since both positive and negative correlations are useful
                    correlation = abs(numeric_col.corr(df_all["correct"]))
                    if not pd.isna(correlation):
                        feature_correlations.append((col, correlation))
            except:
                pass

    # Sort by correlation strength (highest first)
    feature_correlations.sort(key=lambda x: x[1], reverse=True)

    if top_n is not None:
        # Always include required features, then add top correlating features
        required = set(required_features or [])
        top_features = required.copy()

        for feat, corr in feature_correlations:
            if feat not in required:
                top_features.add(feat)
            if len(top_features) >= top_n:
                break

        return list(top_features)

    return [feat for feat, _ in feature_correlations]


def load_predictions_data(features=None):
    """
    Load all past predictions from database.

    Args:
        features: List of feature columns to load. If None, loads default 5 features.

    Returns:
        DataFrame with requested features plus week and correct columns
    """
    if features is None:
        # Default to current formula features
        features = [
            "yards_per_carry_diff",
            "rushing_epa_diff",
            "spread",
            "cover_spread_by",
            "power_ranking_diff_l3",
        ]

    # Always include week and correct
    columns = features + ["week", "correct"]
    columns_str = ", ".join(columns)

    # Load ALL predictions to match how predict.py ranks them
    query = f"""
        SELECT {columns_str}
        FROM past_predictions
        ORDER BY week, game_id
    """
    results = run_query(query)
    df = pd.DataFrame(results)

    # Convert correct to numeric (1 for correct, 0 for incorrect, None for push/null)
    # Keep ALL rows to match ranking behavior in predict.py
    df["correct_numeric"] = df["correct"].apply(
        lambda x: (
            1
            if x == "1" or x == 1 or x == True
            else (0 if x == "0" or x == 0 or x == False else None)
        )
    )

    return df


def calculate_confidence_points(df, confidence_scores):
    """
    Calculate total confidence points for a given confidence score series.
    This matches EXACTLY how predict.py calculates confidence.

    Args:
        df: DataFrame with predictions data (includes ALL games, even pushes)
        confidence_scores: Series of calculated confidence scores (0-100)

    Returns:
        Total confidence points for correct predictions only
    """
    # Use confidence scores directly (already calculated)
    df_test = df.copy()
    df_test["confidence_score"] = confidence_scores

    # Sort ALL games by week and confidence_score (matching predict.py)
    df_test = df_test.sort_values(["week", "confidence_score"])

    # Assign confidence ranks to ALL games within each week (matching predict.py)
    df_test["confidence"] = df_test.groupby("week").cumcount() + 1

    # Sum confidence ONLY for correct predictions (where correct_numeric == 1)
    # This matches: SELECT SUM(confidence) FROM past_predictions WHERE correct='1'
    correct_preds = df_test[df_test["correct_numeric"] == 1]
    total_confidence = correct_preds["confidence"].sum()

    return total_confidence


def test_formula(df, features, weights, scales, fixed_max=10.0):
    """
    Test a specific uncertainty formula configuration.

    Args:
        df: DataFrame with predictions data
        features: List of feature column names in order
        weights: List of weights corresponding to features
        scales: Dict mapping feature names to their scale factors (1 if no scaling)
        fixed_max: Maximum uncertainty value for converting to confidence score

    Returns:
        Total confidence points
    """
    # Calculate uncertainty for this formula
    uncertainty = 0
    for feature, weight in zip(features, weights):
        scale = scales.get(feature, 1)
        if scale != 1:
            uncertainty += abs(df[feature]) / scale * weight
        else:
            uncertainty += abs(df[feature]) * weight

    # Convert uncertainty to confidence score using get_confidence_score
    # Apply to each uncertainty value in the Series
    if isinstance(uncertainty, pd.Series):
        confidence_scores = uncertainty.apply(lambda u: get_confidence_score(None, u))
    else:
        confidence_scores = get_confidence_score(None, uncertainty)

    return calculate_confidence_points(df, confidence_scores)


def test_feature_set(args):
    """
    Test all weight/scale combinations for a single feature set.
    Used for multiprocessing.

    Args:
        args: Tuple of (feat_list, weight_steps, weight_range, scale_factors,
                       granularity, fixed_max, feat_idx, total_sets)
    workers=None,

    Returns:
        Dict with best result for this feature set
    """
    try:
        (
            feat_list,
            weight_steps,
            weight_range,
            scale_factors,
            granularity,
            fixed_max,
            feat_idx,
            total_sets,
        ) = args
    except KeyboardInterrupt:
        # Silently exit on interrupt
        return {"interrupted": True}

    try:
        # Load data for this feature set (each process gets its own DB connection)
        df = load_predictions_data(feat_list)
    except Exception as e:
        return {
            "feature_set": feat_list,
            "error": str(e),
            "best_total": 0,
            "tests_run": 0,
        }

    # Generate scale combinations for these features
    scale_combos = [{}]
    for feat in feat_list:
        if feat in scale_factors and len(scale_factors[feat]) > 1:
            # Generate all combinations
            new_combos = []
            for scale_val in scale_factors[feat]:
                for existing_combo in scale_combos:
                    new_combo = existing_combo.copy()
                    new_combo[feat] = scale_val
                    new_combos.append(new_combo)
            scale_combos = new_combos
        else:
            # No scaling for this feature
            for combo in scale_combos:
                combo[feat] = 1

    # Limit scale combinations for performance
    if granularity == "coarse":
        scale_combos = scale_combos[:50]
    elif granularity == "fine":
        scale_combos = scale_combos[:200]

    # Generate weight combinations
    from itertools import product

    def generate_weight_grid(n_features, n_steps, min_w, max_w):
        """Generate grid of weights that approximately sum to 1.0"""
        weights_list = []
        weight_options = np.linspace(min_w, max_w, n_steps)

        # For small number of features, we can enumerate combinations
        if n_features <= 3:
            for combo in product(weight_options, repeat=n_features):
                if 0.95 <= sum(combo) <= 1.05:
                    # Normalize to exactly 1.0
                    total = sum(combo)
                    normalized = [w / total for w in combo]
                    weights_list.append(normalized)
        else:
            # For more features, use a sampling approach
            # Generate random weight combinations
            np.random.seed(42 + feat_idx)  # Different seed per feature set
            for _ in range(n_steps**3):  # Adjust sample size based on granularity
                # Random weights
                raw_weights = np.random.uniform(min_w, max_w, n_features)
                # Normalize to sum to 1.0
                normalized = raw_weights / raw_weights.sum()
                # Check if all weights are within range
                if all(min_w * 0.8 <= w <= max_w * 1.2 for w in normalized):
                    weights_list.append(normalized.tolist())

        return weights_list

    weight_combos = generate_weight_grid(
        len(feat_list), weight_steps, weight_range[0], weight_range[1]
    )

    # Test all combinations for this feature set
    best_total = 0
    best_config = None
    tests_run = 0

    try:
        for scales in scale_combos:
            for weights in weight_combos:
                tests_run += 1
                try:
                    total = test_formula(df, feat_list, weights, scales, fixed_max)

                    if total > best_total:
                        best_total = total
                        best_config = {
                            "features": feat_list,
                            "weights": weights,
                            "scales": scales,
                            "total": total,
                        }
                except KeyboardInterrupt:
                    # Silently exit on interrupt
                    return {"interrupted": True}
                except:
                    continue
    except KeyboardInterrupt:
        # Silently exit on interrupt
        return {"interrupted": True}

    return {
        "feature_set": feat_list,
        "best_config": best_config,
        "best_total": best_total,
        "tests_run": tests_run,
        "feat_idx": feat_idx,
    }


def optimize_uncertainty_formula(
    granularity="fine",
    features=None,
    min_features=3,
    max_features=5,
    required_features=None,
    weight_range=(0.05, 0.40),
    fixed_max=10.0,
    workers=None,
    verbose=True,
    top_features=12,
):
    """
    Find the optimal uncertainty formula to maximize confidence points.

    This function tests different feature combinations and weight distributions to find
    the formula that maximizes: SELECT SUM(confidence) FROM past_predictions WHERE correct='1'

    The optimization works by:
    1. Loading all past prediction data
    2. Discovering available numeric features from the database
    3. Filtering to top N features by correlation with correct predictions
    4. Testing combinations of features from min_features to max_features
    5. For each formula candidate, recalculating uncertainty and confidence scores
    6. Re-ranking predictions within each week by confidence score
    7. Calculating total confidence points for correct predictions
    8. Finding the formula that achieves the highest total

    The script can be interrupted with Ctrl+C at any time - it will return the best
    formula found so far.

    Args:
        granularity: 'coarse', 'fine', or 'ultra' - search density for weights/scales
        features: List of features to use. If None, tests all combinations from min to max
        min_features: Minimum number of features in each combination (default: 3)
        max_features: Maximum number of features in each combination (default: 5, reduced from 7)
        required_features: List of features that must be included (e.g., ['spread', 'cover_spread_by'])
        weight_range: (min, max) tuple for weight values
        fixed_max: Maximum uncertainty value for scaling (default: 10.0)
        workers: Number of parallel workers. If None, uses cpu_count(). Use 1 to disable multiprocessing.
        verbose: Print progress updates
        top_features: Only consider top N features by correlation with "correct" (default: 12)

    Returns:
        dict with best formula configuration and performance metrics
    """
    if verbose:
        print("=" * 70)
        print("UNCERTAINTY FORMULA OPTIMIZATION")
        print("=" * 70)
        print()

    # Discover available features with correlation filtering
    if verbose:
        print("Analyzing feature correlations with correct predictions...")

    available_features = get_available_features(
        top_n=top_features, required_features=required_features
    )

    if verbose:
        print(f"Selected top {len(available_features)} features by correlation:")
        # Show correlation strength
        all_data = run_query("SELECT * FROM past_predictions WHERE correct != 'push'")
        df_all = pd.DataFrame(all_data)
        df_all["correct"] = df_all["correct"].apply(
            lambda x: 1 if x == "1" or x == 1 or x == True else 0
        )

        for feat in sorted(available_features):
            try:
                numeric_col = pd.to_numeric(df_all[feat], errors="coerce")
                correlation = numeric_col.corr(df_all["correct"])
                print(f"  - {feat:40s} (corr: {correlation:7.4f})")
            except:
                print(f"  - {feat}")
        print()

    # Determine which features to test
    if features is not None:
        # Use explicitly provided features
        feature_sets = [features]
        if verbose:
            print(f"Testing with specific feature set: {features}")
            print()
    else:
        # Test all combinations from min_features to max_features
        from itertools import combinations

        if verbose:
            print(
                f"Testing all combinations from {min_features} to {max_features} features..."
            )
            print(f"Available features: {len(available_features)}")
            if required_features:
                print(f"Required features (always included): {required_features}")
            print()

        # Calculate total combinations
        from math import comb

        if required_features:
            optional_features = [
                f for f in available_features if f not in required_features
            ]
            num_required = len(required_features)
            total_combos = sum(
                comb(len(optional_features), n - num_required)
                for n in range(max(min_features, num_required), max_features + 1)
                if n >= num_required
            )
        else:
            optional_features = available_features
            total_combos = sum(
                comb(len(available_features), n)
                for n in range(min_features, max_features + 1)
            )

        if verbose:
            print(f"Total feature combinations to test: {total_combos:,}")
            print(
                "⚠️  Press Ctrl+C at any time to stop and see the best formula found so far"
            )
            print()

        # Generate all feature combinations
        feature_sets = []
        for n in range(min_features, max_features + 1):
            if required_features:
                num_optional = n - len(required_features)
                if num_optional < 0:
                    continue
                # Generate combinations of optional features
                for optional_combo in combinations(optional_features, num_optional):
                    feature_sets.append(required_features + list(optional_combo))
            else:
                # Generate all combinations of this size
                feature_sets.extend(
                    [list(combo) for combo in combinations(available_features, n)]
                )

        if verbose:
            print(f"Generated {len(feature_sets)} feature combinations")
            print()

    current_total = get_current_confidence_total()

    if verbose:
        print(f"Current database total: {current_total} confidence points")
        print()

    # Set search density based on granularity
    if granularity == "coarse":
        weight_steps = 6
    elif granularity == "fine":
        weight_steps = 10
    elif granularity == "ultra":
        weight_steps = 15
    else:
        raise ValueError("granularity must be 'coarse', 'fine', or 'ultra'")

    # Determine number of workers
    if workers is None:
        workers = cpu_count()

    if verbose:
        print(f"Search granularity: {granularity} (weight steps: {weight_steps})")
        print(f"Parallel workers: {workers}")
        print()

        # Estimate total tests
        avg_weight_combos = (
            weight_steps**3 if len(feature_sets[0]) > 3 else weight_steps**2
        )
        avg_scale_combos = 50 if granularity == "coarse" else 200
        total_tests_estimate = len(feature_sets) * avg_weight_combos * avg_scale_combos
        print(f"Estimated total tests: ~{total_tests_estimate:,}")
        print()
        print("Searching for optimal formula...")
        print()

    best_total = current_total
    best_config = None
    total_tests_run = 0
    improvements_found = 0

    # Common scale factors for known features
    scale_factors = {
        "rushing_epa_diff": [1, 40, 45, 50, 55, 60],
        "passing_epa_diff": [1, 40, 45, 50, 55, 60],
        "spread": [1, 7, 8, 9, 10, 11, 12],
        "spread_diff": [1, 5, 10, 15],
        "power_ranking_diff": [1, 5, 10],
        "power_ranking_diff_l3": [1, 5, 10],
        "avg_weekly_point_diff": [1, 5, 10, 15],
        "avg_weekly_point_diff_l3": [1, 5, 10, 15],
        "avg_margin_of_victory_diff": [1, 5, 10, 15],
        "avg_margin_of_victory_diff_l3": [1, 5, 10, 15],
        "predicted_diff": [1, 5, 10, 15],
    }

    # Prepare arguments for parallel processing
    test_args = [
        (
            feat_list,
            weight_steps,
            weight_range,
            scale_factors,
            granularity,
            fixed_max,
            feat_idx,
            len(feature_sets),
        )
        for feat_idx, feat_list in enumerate(feature_sets)
    ]

    # Process feature sets in parallel
    start_time = time()
    pool = None
    try:
        if workers == 1:
            # Single-threaded mode (useful for debugging)
            results = []
            for args in test_args:
                if verbose:
                    elapsed = time() - start_time
                    progress = (args[6] + 1) / args[7] * 100
                    rate = (args[6] + 1) / elapsed if elapsed > 0 else 0
                    eta = (args[7] - args[6] - 1) / rate if rate > 0 else 0
                    print(
                        f"[{progress:5.1f}%] Testing feature set {args[6] + 1}/{args[7]}: {args[0]}"
                    )
                    print(
                        f"         Elapsed: {elapsed/60:.1f}m | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m"
                    )
                result = test_feature_set(args)
                results.append(result)

                # Update best if needed
                if result["best_config"] and result["best_total"] > best_total:
                    improvements_found += 1
                    best_total = result["best_total"]
                    best_config = result["best_config"]

                    if verbose:
                        improvement = best_total - current_total
                        print(
                            f"🎯 NEW BEST: {best_total:.0f} points (+{improvement:.0f})"
                        )
                        for i, feat in enumerate(best_config["features"]):
                            scale_str = (
                                f"/{best_config['scales'][feat]}"
                                if best_config["scales"][feat] != 1
                                else ""
                            )
                            print(
                                f"    {feat}{scale_str} * {best_config['weights'][i]:.3f}"
                            )

                total_tests_run += result["tests_run"]
        else:
            # Multi-processing mode
            pool = Pool(processes=workers, initializer=worker_init)
            if verbose:
                print(
                    f"Processing {len(test_args)} feature sets across {workers} workers..."
                )
                print()

            # Use imap_unordered for better performance
            completed = 0
            last_update = time()
            last_completion = start_time
            recent_times = []  # Track recent completion times for moving average

            for result in pool.imap_unordered(test_feature_set, test_args, chunksize=1):
                completed += 1
                current_time = time()
                set_time = current_time - last_completion
                last_completion = current_time

                # Track time for moving average rate calculation (last 20 completions)
                recent_times.append(current_time)
                if len(recent_times) > 20:
                    recent_times.pop(0)

                # Check if worker was interrupted
                if result.get("interrupted"):
                    continue

                # Check for errors
                if "error" in result:
                    if verbose:
                        print(
                            f"⚠️  Skipped feature set {result['feature_set']}: {result['error']}"
                        )
                    continue

                # Update best if needed
                if result["best_config"] and result["best_total"] > best_total:
                    improvements_found += 1
                    best_total = result["best_total"]
                    best_config = result["best_config"]

                    if verbose:
                        improvement = best_total - current_total
                        print(
                            f"🎯 NEW BEST: {best_total:.0f} points (+{improvement:.0f}) | Set took {set_time/60:.1f}m"
                        )
                        for i, feat in enumerate(best_config["features"]):
                            scale_str = (
                                f"/{best_config['scales'][feat]}"
                                if best_config["scales"][feat] != 1
                                else ""
                            )
                            print(
                                f"    {feat}{scale_str} * {best_config['weights'][i]:.3f}"
                            )

                total_tests_run += result["tests_run"]

                # Progress update - every 1 completion or every 10 seconds
                if verbose and (
                    completed % 1 == 0 or (current_time - last_update) >= 10
                ):
                    elapsed = current_time - start_time
                    pct = (completed / len(test_args)) * 100

                    # Calculate rate using moving average of recent completions
                    if len(recent_times) >= 2:
                        recent_elapsed = recent_times[-1] - recent_times[0]
                        rate = (
                            (len(recent_times) - 1) / recent_elapsed
                            if recent_elapsed > 0
                            else 0
                        )
                    else:
                        rate = completed / elapsed if elapsed > 0 else 0

                    eta = (len(test_args) - completed) / rate if rate > 0 else 0

                    # Show per-set time for slow sets (> 30 seconds)
                    set_time_str = f" | Set: {set_time:.0f}s" if set_time > 30 else ""

                    print(
                        f"[{pct:5.1f}%] {completed:,}/{len(test_args):,} sets | "
                        f"Best: {best_total:.0f} | "
                        f"Rate: {rate:.2f}/s | "
                        f"ETA: {eta/60:.0f}m{set_time_str}"
                    )
                    last_update = current_time

    except KeyboardInterrupt:
        if verbose:
            elapsed = time() - start_time
            print()
            print(f"⚠️  Interrupted after {elapsed/60:.1f} minutes!")
            print("Terminating worker processes (ignore tracebacks below)...")

        # Terminate the pool if it exists
        if pool is not None:
            pool.terminate()
            pool.join()

        if verbose:
            print()
            print("Returning best formula found so far...")
            print()

    finally:
        # Clean up pool if it was created
        if pool is not None:
            pool.close()
            pool.join()

    if verbose:
        elapsed = time() - start_time
        print()
        print("=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Feature sets tested: {len(feature_sets)}")
        print(f"Total formulas tested: {total_tests_run:,}")
        print(f"Improvements found: {improvements_found}")
        print()

    if best_config:
        if verbose:
            print("BEST FORMULA FOUND:")
            print(f"Total confidence points: {best_config['total']:.0f}")
            print(f"Improvement: +{best_config['total'] - current_total:.0f} points")
            print(f"Gap to 1426 benchmark: {1426 - best_config['total']:.0f} points")
            print()
            print(
                "⚠️  NOTE: This total is based on the current data in past_predictions table."
            )
            print(
                "    When you update get_confidence_score() and re-run predictions, the actual"
            )
            print(
                "    total may differ if training_data has changed (new games, updated features)."
            )
            print(
                "    Run 'nfl model predict past' first to sync past_predictions with training_data."
            )
            print()
            print("Features:")
            for i, feat in enumerate(best_config["features"]):
                scale = best_config["scales"][feat]
                weight = best_config["weights"][i]
                scale_str = f" / {scale}" if scale != 1 else ""
                print(f"  - {feat}{scale_str} * {weight:.3f}")
            print()
            print("Copy this into get_confidence_score() in predict.py:")
            print("-" * 70)
            print("uncertainty = (")
            for i, feat in enumerate(best_config["features"]):
                scale = best_config["scales"][feat]
                weight = best_config["weights"][i]

                # Format for predict.py get_confidence_score() function
                if i == 0:
                    prefix = "    "
                else:
                    prefix = "    + "

                # Note: 'spread' in DB is already absolute value, no abs() needed
                # But cover_spread_by and other features need abs()
                if feat == "spread":
                    abs_wrap = ""  # Already absolute in DB
                else:
                    abs_wrap = "abs"

                var_name = f'game_data["{feat}"]'

                if scale != 1:
                    if abs_wrap:
                        print(
                            f"{prefix}{abs_wrap}({var_name}) / {scale} * {weight:.3f}"
                        )
                    else:
                        print(f"{prefix}{var_name} / {scale} * {weight:.3f}")
                else:
                    if abs_wrap:
                        print(f"{prefix}{abs_wrap}({var_name}) * {weight:.3f}")
                    else:
                        print(f"{prefix}{var_name} * {weight:.3f}")
            print(")")
            print("-" * 70)

        return {
            "best_config": best_config,
            "current_total": current_total,
            "improvement": best_config["total"] - current_total,
            "tests_run": total_tests_run,
        }
    else:
        if verbose:
            print(
                "No improvements found. Current formula is optimal within tested ranges."
            )
        return {
            "best_config": None,
            "current_total": current_total,
            "improvement": 0,
            "tests_run": total_tests_run,
        }


def verify_calculation():
    """
    Verify that our calculation method works correctly.

    Note: The database has pre-calculated confidence values based on a specific formula.
    This function shows what total we'd get if we recalculate using the current formula.
    The totals may differ because the database values were calculated at a different time
    or with different parameters.
    """
    print("=" * 70)
    print("CALCULATION VERIFICATION")
    print("=" * 70)
    print()

    # Get current database total
    db_total = get_current_confidence_total()
    print(f"Current database total: {db_total} confidence points")
    print()

    # Get all available features (not just defaults)
    all_features = get_available_features()

    # Load data with all features and recalculate using current formula
    df = load_predictions_data(features=all_features)
    print(f"Loaded {len(df)} predictions for recalculation")
    print()

    # Current formula from predict.py get_confidence_score() function
    print("Testing with current formula from predict.py:")
    # Calculate confidence scores using the actual function from predict.py
    confidence_scores = pd.Series([0.0] * len(df), index=df.index)
    for idx, row in df.iterrows():
        game_data = row.to_dict()
        confidence_scores[idx] = get_confidence_score(game_data)

    calculated_total = calculate_confidence_points(df, confidence_scores)
    print(f"Recalculated total: {calculated_total:.0f} confidence points")
    print()

    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("The database total and recalculated total may differ because:")
    print("1. The database was populated with a previous formula")
    print("2. The optimization will find formulas that maximize the recalculated total")
    print("3. After optimization, you should re-run predictions to update the database")
    print("=" * 70)
    print()


if __name__ == "__main__":
    # Run verification first
    verify_calculation()

    # Run optimization with fine granularity
    results = optimize_uncertainty_formula(granularity="fine", verbose=True)


def run_optimization(
    granularity="coarse",
    min_features=3,
    max_features=5,
    workers=None,
    verify=False,
    top_features=12,
):
    """
    Main entry point for running uncertainty formula optimization.

    Args:
        granularity: 'coarse', 'fine', or 'ultra' (default: coarse for speed)
        min_features: Minimum number of features in each combination
        max_features: Maximum number of features in each combination (default: 5, reduced from 7)
        workers: Number of parallel workers (None = auto-detect)
        verify: Whether to run verification first
        top_features: Only consider top N features by correlation (default: 12)
    """
    if verify:
        verify_calculation()

    print(f"Starting optimization with {granularity} granularity...")
    print(f"Testing all combinations from {min_features} to {max_features} features...")
    print(
        f"Pre-filtering to top {top_features} features by correlation with correct predictions..."
    )
    print("⚠️  This will test many combinations and may take a long time!")
    print("⚠️  Press Ctrl+C at any time to stop and see the best formula found so far")
    if workers is None or workers > 1:
        print(
            "⚠️  NOTE: On Windows, Ctrl+C may show worker tracebacks. Use --workers 1 for clean interrupts."
        )
    print()

    # Set required features that should always be included
    required_features = ["spread", "cover_spread_by"]

    try:
        results = optimize_uncertainty_formula(
            granularity=granularity,
            min_features=min_features,
            max_features=max_features,
            required_features=required_features,
            workers=workers,
            verbose=True,
            top_features=top_features,
        )
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("OPTIMIZATION INTERRUPTED BY USER")
        print("=" * 70)
        print("Returning best formula found so far...")
        print()
        # The best_config will be in the closure, but we need to handle this differently
        # For now, just re-raise and let the outer function handle it
        raise

    if results["best_config"]:
        print()
        print("=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print(
            "1. Copy the uncertainty formula above into predict.py get_confidence_score() function"
        )
        print("2. Run: python nfl.py model predict past --year 2025")
        print("   This will recalculate all predictions with the new formula")
        print(
            "3. Verify the new total: SELECT SUM(confidence) FROM past_predictions WHERE correct='1'"
        )
        print("   The database total should now match the optimized total shown above")
        print("=" * 70)
    else:
        print()
        print("No better formula found within the tested parameter ranges.")
        print("You may want to:")
        print("  - Try 'fine' or 'ultra' granularity for more thorough search")
        print("  - Increase --top-features to consider more features")
        print("  - Adjust --min-features and --max-features to test different ranges")

    return results
