import pandas as pd
from datetime import datetime
from sqlalchemy import text

from src.helpers.database_helpers import get_db_engine
from src.reports.nfl_past_prediction_report import lookup_accuracy_metric


def calculate_metric_reliability(model_baseline=62.9):
    """
    Calculate DIRECTIONAL reliability scores for metrics.

    For each metric, calculates:
    - Positive Indicator Reliability: Of all CORRECT predictions, how often did this metric appear as positive?
                                       Of all INCORRECT predictions, how often did it appear as positive?
    - Negative Indicator Reliability: Of all CORRECT predictions, how often did it appear as negative?
                                       Of all INCORRECT predictions, how often did it appear as negative?

    Returns:
        dict: Metric reliability data keyed by metric_type with separate scores for positive/negative indicators
    """
    engine = get_db_engine()

    # Load all past predictions with outcomes
    past_predictions_query = """
        SELECT * FROM past_predictions
        WHERE season = 2025 AND correct != 'push'
    """
    predictions_df = pd.read_sql(past_predictions_query, engine)
    predictions_df["correct"] = pd.to_numeric(predictions_df["correct"])

    if predictions_df.empty:
        return {}

    # Load all metric accuracy data to categorize buckets
    metrics_query = """
        SELECT metric_type, category, subcategory, correct, total, accuracy
        FROM prediction_accuracy_metrics
        WHERE total >= 5
        ORDER BY metric_type, accuracy DESC
    """
    metrics_df = pd.read_sql(metrics_query, engine)

    if metrics_df.empty:
        return {}

    # Build bucket categorization lookup
    bucket_categories = {}
    for _, row in metrics_df.iterrows():
        metric_type = row["metric_type"]
        category = row["category"]
        accuracy = row["accuracy"]

        if metric_type not in bucket_categories:
            bucket_categories[metric_type] = {}

        # Categorize bucket
        if accuracy >= model_baseline:
            bucket_categories[metric_type][category] = "positive"
        elif accuracy >= (model_baseline - 5):
            bucket_categories[metric_type][category] = "warning"
        else:
            bucket_categories[metric_type][category] = "negative"

    metric_stats = {}
    total_correct = predictions_df["correct"].sum()
    total_incorrect = len(predictions_df) - total_correct

    # For each metric, analyze its directional reliability
    for metric_type in bucket_categories.keys():
        stats = {
            "metric_name": metric_type.replace("_", " ").title(),
            # Positive indicator stats
            "positive_in_correct_predictions": 0,
            "positive_in_incorrect_predictions": 0,
            "positive_indicator_reliability": 0,
            "positive_precision": 0,  # When metric is positive, % that are correct
            # Negative indicator stats
            "negative_in_correct_predictions": 0,
            "negative_in_incorrect_predictions": 0,
            "negative_indicator_reliability": 0,
            "negative_precision": 0,  # When metric is negative, % that are incorrect
            # Overall stats
            "total_appearances": 0,
            "positive_appearances": 0,
            "negative_appearances": 0,
        }

        # Analyze each prediction
        for _, pred in predictions_df.iterrows():
            # Get metric value and determine its bucket/category
            # This is simplified - in reality we'd need to call lookup_accuracy_metric
            # For now, we'll use a proxy: analyze the metric buckets
            pass  # TODO: Need to actually look up which bucket each prediction fell into

        # For now, use the simpler bucket-based approach but calculate directional metrics
        positive_buckets = [
            cat
            for cat, type_ in bucket_categories[metric_type].items()
            if type_ == "positive"
        ]
        negative_buckets = [
            cat
            for cat, type_ in bucket_categories[metric_type].items()
            if type_ == "negative"
        ]

        # Get stats for positive buckets
        positive_bucket_data = metrics_df[
            (metrics_df["metric_type"] == metric_type)
            & (metrics_df["category"].isin(positive_buckets))
        ]

        if not positive_bucket_data.empty:
            stats["positive_appearances"] = int(positive_bucket_data["total"].sum())
            stats["positive_in_correct_predictions"] = int(
                positive_bucket_data["correct"].sum()
            )
            stats["positive_in_incorrect_predictions"] = (
                stats["positive_appearances"] - stats["positive_in_correct_predictions"]
            )

            # Positive indicator reliability: how often does it appear in wins vs losses?
            if total_correct > 0 and total_incorrect > 0:
                positive_rate_in_wins = (
                    stats["positive_in_correct_predictions"] / total_correct
                )
                positive_rate_in_losses = (
                    stats["positive_in_incorrect_predictions"] / total_incorrect
                )
                stats["positive_indicator_reliability"] = (
                    (
                        positive_rate_in_wins
                        / (positive_rate_in_wins + positive_rate_in_losses)
                    )
                    * 100
                    if (positive_rate_in_wins + positive_rate_in_losses) > 0
                    else 50
                )

            # Precision: when this metric is positive, what % are correct?
            stats["positive_precision"] = (
                (
                    stats["positive_in_correct_predictions"]
                    / stats["positive_appearances"]
                    * 100
                )
                if stats["positive_appearances"] > 0
                else 0
            )

        # Get stats for negative buckets
        negative_bucket_data = metrics_df[
            (metrics_df["metric_type"] == metric_type)
            & (metrics_df["category"].isin(negative_buckets))
        ]

        if not negative_bucket_data.empty:
            stats["negative_appearances"] = int(negative_bucket_data["total"].sum())
            stats["negative_in_correct_predictions"] = int(
                negative_bucket_data["correct"].sum()
            )
            stats["negative_in_incorrect_predictions"] = (
                stats["negative_appearances"] - stats["negative_in_correct_predictions"]
            )

            # Negative indicator reliability: how often does it appear in losses vs wins?
            if total_correct > 0 and total_incorrect > 0:
                negative_rate_in_losses = (
                    stats["negative_in_incorrect_predictions"] / total_incorrect
                )
                negative_rate_in_wins = (
                    stats["negative_in_correct_predictions"] / total_correct
                )
                stats["negative_indicator_reliability"] = (
                    (
                        negative_rate_in_losses
                        / (negative_rate_in_losses + negative_rate_in_wins)
                    )
                    * 100
                    if (negative_rate_in_losses + negative_rate_in_wins) > 0
                    else 50
                )

            # Precision: when this metric is negative, what % are incorrect?
            stats["negative_precision"] = (
                (
                    stats["negative_in_incorrect_predictions"]
                    / stats["negative_appearances"]
                    * 100
                )
                if stats["negative_appearances"] > 0
                else 0
            )

        stats["total_appearances"] = (
            stats["positive_appearances"] + stats["negative_appearances"]
        )

        # Overall reliability score (average of directional reliabilities weighted by frequency)
        total_weighted = (
            stats["positive_indicator_reliability"] * stats["positive_appearances"]
            + stats["negative_indicator_reliability"] * stats["negative_appearances"]
        )
        stats["reliability_score"] = (
            (total_weighted / stats["total_appearances"])
            if stats["total_appearances"] > 0
            else 0
        )

        metric_stats[metric_type] = stats

    return metric_stats


def is_metric_meaningful(metric_type, min_variance=10.0, min_sample_size=5):
    """
    Two-stage filtering to determine if a metric is meaningful for outlier detection.

    Stage 1: Remove buckets with insufficient sample sizes
    Stage 2: Check if remaining buckets have meaningful variance

    This prevents metrics where variance is artificially inflated by unreliable
    low-sample buckets from being used.

    Example:
        Buckets: [65% (50 games), 62% (48 games), 61% (45 games), 50% (2 games)]
        - Remove 50% bucket (2 games < 5)
        - Remaining: 65%, 62%, 61% → variance = 4% < 10%
        - Result: Metric is NOT meaningful (filtered out)

    Args:
        metric_type: The metric type to evaluate (e.g., 'primetime', 'pick_location')
        min_variance: Minimum accuracy difference between best and worst bucket (default: 10%)
        min_sample_size: Minimum number of games in each bucket (default: 5)

    Returns:
        bool: True if metric is meaningful, False otherwise
    """
    engine = get_db_engine()

    query = text(
        """
        SELECT category, accuracy, total
        FROM prediction_accuracy_metrics
        WHERE metric_type = :metric_type
    """
    )

    with engine.connect() as conn:
        result = conn.execute(query, {"metric_type": metric_type}).fetchall()

    if not result:
        return False

    # Stage 1: Filter out buckets with insufficient sample sizes
    usable_buckets = [(row[1], row[2]) for row in result if row[2] >= min_sample_size]

    # Need at least 2 buckets to measure variance
    if len(usable_buckets) < 2:
        return False

    # Stage 2: Check variance among remaining usable buckets
    usable_accuracies = [bucket[0] for bucket in usable_buckets]
    max_accuracy = max(usable_accuracies)
    min_accuracy = min(usable_accuracies)
    variance = max_accuracy - min_accuracy

    return variance >= min_variance


def load_future_predictions():
    """Load all future predictions from the database."""
    engine = get_db_engine()

    # First, get the list of actual columns in the table
    with engine.connect() as conn:
        # Get table info to see what columns exist
        table_info = conn.execute(
            text("PRAGMA table_info(future_predictions)")
        ).fetchall()
        existing_columns = [col[1] for col in table_info]

    # Define all possible columns we want (in order of preference)
    desired_columns = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "spread",
        "spread_favorite",
        "div_game",
        "predicted_winner",
        "cover_spread_by",
        "power_ranking_diff",
        "power_ranking_diff_l3",
        "win_pct_diff",
        "avg_weekly_point_diff",
        "avg_weekly_point_diff_l3",
        "avg_margin_of_victory_diff",
        "avg_margin_of_victory_diff_l3",
        "passing_epa_diff",
        "rushing_epa_diff",
        "spread_performance_diff",
        "sacks_suffered_avg_diff",
        "adj_overall_rank_diff",
        "home_qb_changed",
        "away_qb_changed",
        "confidence",
        "completion_pct_diff",
        "yards_per_attempt_diff",
        "yards_per_carry_diff",
        "turnover_diff",
        "rest_diff",
        "spread_diff",
        "rushing_yards_diff",
        "def_rushing_yards_diff",
        "passing_yards_diff",
    ]

    # Only select columns that actually exist
    columns_to_select = [col for col in desired_columns if col in existing_columns]

    # Build dynamic query
    query = text(
        f"""
        SELECT {', '.join(columns_to_select)}
        FROM future_predictions
        ORDER BY week, confidence DESC
    """
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        predictions = result.fetchall()
        columns = result.keys()

    df = pd.DataFrame(predictions, columns=columns)
    return df


def analyze_prediction_metrics(prediction, reliability_data=None):
    """
    Analyze a single prediction and return all relevant accuracy metrics.

    Two-stage filtering approach:
    1. Metric-level: Skip metrics where usable buckets (≥5 games) lack meaningful variance (≥10%)
    2. Bucket-level: For valid metrics, skip individual buckets with <5 games

    Thresholds:
    - Positive indicators: ≥60% accuracy
    - Warning indicators: Below model baseline but not significantly
    - Negative indicators: <50% accuracy

    Args:
        prediction: Dict containing prediction data
        reliability_data: Dict of metric reliability scores (optional, calculated if None)

    Returns:
        dict with metrics categorized as positive_indicators, warning_indicators, negative_indicators,
        plus 'stage2_filtered' list showing which metrics were skipped due to insufficient bucket samples
    """
    # Calculate reliability data if not provided
    if reliability_data is None:
        reliability_data = calculate_metric_reliability()

    metrics = {
        "positive_indicators": [],  # accuracy >= model baseline
        "warning_indicators": [],  # model baseline > accuracy >= baseline - 5%
        "negative_indicators": [],  # accuracy < baseline - 5%
        "stage2_filtered": [],  # metrics that passed Stage 1 but filtered in Stage 2 for this game
    }

    # Two-stage filtering:
    # - is_metric_meaningful() checks if usable buckets have variance (Stage 1)
    # - _categorize_metric() checks if specific bucket has enough samples (Stage 2)

    # 1. Spread magnitude
    if is_metric_meaningful("spread_magnitude"):
        spread_result = lookup_accuracy_metric(
            "spread_magnitude", abs(prediction["spread"])
        )
        if spread_result:
            metrics = _categorize_metric(
                metrics, "Spread Magnitude", spread_result, reliability_data
            )

    # 2. Margin magnitude
    if is_metric_meaningful("margin_magnitude"):
        margin_result = lookup_accuracy_metric(
            "margin_magnitude", abs(prediction["cover_spread_by"])
        )
        if margin_result:
            metrics = _categorize_metric(
                metrics, "Predicted Margin", margin_result, reliability_data
            )

    # 3. Power ranking differential
    if pd.notna(prediction["power_ranking_diff"]) and is_metric_meaningful(
        "power_ranking_diff"
    ):
        power_result = lookup_accuracy_metric(
            "power_ranking_diff", abs(prediction["power_ranking_diff"])
        )
        if power_result:
            metrics = _categorize_metric(
                metrics, "Power Ranking Differential", power_result, reliability_data
            )

    # 4. Adjusted overall rank differential
    if pd.notna(prediction["adj_overall_rank_diff"]) and is_metric_meaningful(
        "adj_overall_rank_diff"
    ):
        rank_result = lookup_accuracy_metric(
            "adj_overall_rank_diff", abs(prediction["adj_overall_rank_diff"])
        )
        if rank_result:
            metrics = _categorize_metric(
                metrics, "Adjusted Rank Differential", rank_result, reliability_data
            )

    # 5. Predicted team
    if is_metric_meaningful("predicted_team"):
        team_result = lookup_accuracy_metric(
            "predicted_team", prediction["predicted_winner"]
        )
        if team_result:
            metrics = _categorize_metric(
                metrics,
                f'Team: {prediction["predicted_winner"]}',
                team_result,
                reliability_data,
            )

    # 6. Pick location (home vs away)
    if is_metric_meaningful("pick_location"):
        is_home_pick = prediction["predicted_winner"] == prediction["home_team"]
        location = "Home Team" if is_home_pick else "Away Team"
        location_result = lookup_accuracy_metric("pick_location", location)
        if location_result:
            metrics = _categorize_metric(
                metrics, "Pick Location", location_result, reliability_data
            )

    # 7. Favorite vs Underdog
    if is_metric_meaningful("favorite_underdog"):
        is_favorite = prediction["predicted_winner"] == prediction["spread_favorite"]
        pick_type = "Favorite" if is_favorite else "Underdog"
        fav_result = lookup_accuracy_metric("favorite_underdog", pick_type)
        if fav_result:
            metrics = _categorize_metric(
                metrics, "Favorite/Underdog", fav_result, reliability_data
            )

    # 8. Division game
    if is_metric_meaningful("division_game"):
        game_type = "Division Game" if prediction["div_game"] == 1 else "Non-Division"
        div_result = lookup_accuracy_metric("division_game", game_type)
        if div_result:
            metrics = _categorize_metric(
                metrics, "Division Game", div_result, reliability_data
            )

    # 9. Prime time (if data available)
    # Note: This requires primetime data in prediction - would need to be added
    # Placeholder for when that data is available

    # 10. Rest differential (if data available)
    # Note: This requires rest_diff data in prediction - would need to be added
    # Placeholder for when that data is available

    # 11. QB changes (simplified - any change vs no change)
    if is_metric_meaningful("qb_change_any"):
        has_qb_change = (
            prediction["home_qb_changed"] == "Y" or prediction["away_qb_changed"] == "Y"
        )
        qb_any_category = "QB Change (Any)" if has_qb_change else "No QB Change"

        qb_any_result = lookup_accuracy_metric("qb_change_any", qb_any_category)
        if qb_any_result:
            metrics = _categorize_metric(
                metrics, "QB Change (Any vs None)", qb_any_result, reliability_data
            )

    # 12. Close game
    if is_metric_meaningful("close_game"):
        close_category = (
            "Close (<3 pts)" if abs(prediction["spread"]) < 3 else "Not Close (≥3 pts)"
        )
        close_result = lookup_accuracy_metric("close_game", close_category)
        if close_result:
            metrics = _categorize_metric(
                metrics, "Close Game", close_result, reliability_data
            )

    # 13. Recent form
    if pd.notna(prediction.get("avg_weekly_point_diff_l3")) and is_metric_meaningful(
        "recent_form"
    ):
        strong_recent_form = abs(prediction["avg_weekly_point_diff_l3"]) > 10
        form_category = (
            "Strong Recent Form (L3 diff >10)"
            if strong_recent_form
            else "Normal Recent Form"
        )
        form_result = lookup_accuracy_metric("recent_form", form_category)
        if form_result:
            metrics = _categorize_metric(
                metrics, "Recent Form", form_result, reliability_data
            )

    # 14. EPA mismatch
    if (
        pd.notna(prediction.get("passing_epa_diff"))
        and pd.notna(prediction.get("rushing_epa_diff"))
        and is_metric_meaningful("epa_mismatch")
    ):
        has_mismatch = (
            prediction["passing_epa_diff"] > 50 and prediction["rushing_epa_diff"] < -50
        ) or (
            prediction["passing_epa_diff"] < -50 and prediction["rushing_epa_diff"] > 50
        )
        epa_category = "Pass/Rush EPA Mismatch" if has_mismatch else "EPA Aligned"
        epa_result = lookup_accuracy_metric("epa_mismatch", epa_category)
        if epa_result:
            metrics = _categorize_metric(
                metrics, "EPA Alignment", epa_result, reliability_data
            )

    # 15. Weather (if data available)
    # Note: This requires weather data in prediction - would need to be added
    # Placeholder for when that data is available

    # TOP 10 MODEL FEATURES

    # 16. Power ranking diff L3 (HIGHEST IMPORTANCE FEATURE - #1 at 9.06%)
    if pd.notna(prediction.get("power_ranking_diff_l3")) and is_metric_meaningful(
        "power_ranking_diff_l3"
    ):
        pr_l3_result = lookup_accuracy_metric(
            "power_ranking_diff_l3", abs(prediction["power_ranking_diff_l3"])
        )
        if pr_l3_result:
            metrics = _categorize_metric(
                metrics,
                "⭐ Power Ranking L3 (Top Feature)",
                pr_l3_result,
                reliability_data,
            )

    # 17. Average margin of victory differential (#3 at 6.49%)
    if pd.notna(prediction.get("avg_margin_of_victory_diff")) and is_metric_meaningful(
        "avg_margin_of_victory_diff"
    ):
        mov_result = lookup_accuracy_metric(
            "avg_margin_of_victory_diff", abs(prediction["avg_margin_of_victory_diff"])
        )
        if mov_result:
            metrics = _categorize_metric(
                metrics, "Avg Margin of Victory Diff", mov_result, reliability_data
            )

    # 18. Average margin of victory differential L3 (#4 at 4.84%)
    if pd.notna(
        prediction.get("avg_margin_of_victory_diff_l3")
    ) and is_metric_meaningful("avg_margin_of_victory_diff_l3"):
        mov_l3_result = lookup_accuracy_metric(
            "avg_margin_of_victory_diff_l3",
            abs(prediction["avg_margin_of_victory_diff_l3"]),
        )
        if mov_l3_result:
            metrics = _categorize_metric(
                metrics, "Avg MOV Diff L3", mov_l3_result, reliability_data
            )

    # 19. Average weekly point differential (#5 at 4.47%)
    if pd.notna(prediction.get("avg_weekly_point_diff")) and is_metric_meaningful(
        "avg_weekly_point_diff"
    ):
        weekly_pt_result = lookup_accuracy_metric(
            "avg_weekly_point_diff", abs(prediction["avg_weekly_point_diff"])
        )
        if weekly_pt_result:
            metrics = _categorize_metric(
                metrics,
                "⚡ Avg Weekly Point Diff L3 (OPTIMIZED)",
                weekly_pt_result,
                reliability_data,
            )

    # 20. Spread performance differential (#7 at 4.11%)
    if pd.notna(prediction.get("spread_performance_diff")) and is_metric_meaningful(
        "spread_performance_diff"
    ):
        spread_perf_result = lookup_accuracy_metric(
            "spread_performance_diff", abs(prediction["spread_performance_diff"])
        )
        if spread_perf_result:
            metrics = _categorize_metric(
                metrics,
                "⚡ Spread Performance Diff (OPTIMIZED)",
                spread_perf_result,
                reliability_data,
            )

    # 21. Sacks suffered differential (#10 at 3.54%)
    if pd.notna(prediction.get("sacks_suffered_avg_diff")) and is_metric_meaningful(
        "sacks_suffered_avg_diff"
    ):
        sacks_result = lookup_accuracy_metric(
            "sacks_suffered_avg_diff", abs(prediction["sacks_suffered_avg_diff"])
        )
        if sacks_result:
            metrics = _categorize_metric(
                metrics, "Sacks Suffered Diff", sacks_result, reliability_data
            )

    # 22. Completion percentage differential
    if pd.notna(prediction.get("completion_pct_diff")) and is_metric_meaningful(
        "completion_pct_diff"
    ):
        comp_pct_result = lookup_accuracy_metric(
            "completion_pct_diff", abs(prediction["completion_pct_diff"])
        )
        if comp_pct_result:
            metrics = _categorize_metric(
                metrics, "Completion % Diff", comp_pct_result, reliability_data
            )

    # 23. Yards per attempt differential
    if pd.notna(prediction.get("yards_per_attempt_diff")) and is_metric_meaningful(
        "yards_per_attempt_diff"
    ):
        ypa_result = lookup_accuracy_metric(
            "yards_per_attempt_diff", abs(prediction["yards_per_attempt_diff"])
        )
        if ypa_result:
            metrics = _categorize_metric(
                metrics,
                "⚡ Yards Per Attempt Differential (OPTIMIZED)",
                ypa_result,
                reliability_data,
            )

    # 24. Yards per carry differential
    if pd.notna(prediction.get("yards_per_carry_diff")) and is_metric_meaningful(
        "yards_per_carry_diff"
    ):
        ypc_result = lookup_accuracy_metric(
            "yards_per_carry_diff", abs(prediction["yards_per_carry_diff"])
        )
        if ypc_result:
            metrics = _categorize_metric(
                metrics,
                "⚡ Yards Per Carry Differential (OPTIMIZED)",
                ypc_result,
                reliability_data,
            )

    # 25. Turnover differential
    if pd.notna(prediction.get("turnover_diff")) and is_metric_meaningful(
        "turnover_diff"
    ):
        turnover_result = lookup_accuracy_metric(
            "turnover_diff", prediction["turnover_diff"]
        )
        if turnover_result:
            metrics = _categorize_metric(
                metrics, "Turnover Differential", turnover_result, reliability_data
            )

    # 26. Spread differential
    if pd.notna(prediction.get("spread_diff")) and is_metric_meaningful("spread_diff"):
        spread_diff_result = lookup_accuracy_metric(
            "spread_diff", prediction["spread_diff"]
        )
        if spread_diff_result:
            metrics = _categorize_metric(
                metrics,
                "Spread Differential (Sportsbook Agreement)",
                spread_diff_result,
                reliability_data,
            )

    # 27. Rushing yards differential (TOP OUTLIER METRIC - 30% variance)
    if pd.notna(prediction.get("rushing_yards_diff")) and is_metric_meaningful(
        "rushing_yards_diff"
    ):
        rushing_yards_result = lookup_accuracy_metric(
            "rushing_yards_diff", prediction["rushing_yards_diff"]
        )
        if rushing_yards_result:
            metrics = _categorize_metric(
                metrics,
                "🏃 Rushing Yards Differential (⚠️ TOP OUTLIER)",
                rushing_yards_result,
                reliability_data,
            )

    # 28. Defensive rushing yards differential
    if pd.notna(prediction.get("def_rushing_yards_diff")) and is_metric_meaningful(
        "def_rushing_yards_diff"
    ):
        def_rushing_result = lookup_accuracy_metric(
            "def_rushing_yards_diff", prediction["def_rushing_yards_diff"]
        )
        if def_rushing_result:
            metrics = _categorize_metric(
                metrics,
                "🛡️ Defensive Rushing Yards Differential",
                def_rushing_result,
                reliability_data,
            )

    # 29. Passing yards differential
    if pd.notna(prediction.get("passing_yards_diff")) and is_metric_meaningful(
        "passing_yards_diff"
    ):
        passing_yards_result = lookup_accuracy_metric(
            "passing_yards_diff", prediction["passing_yards_diff"]
        )
        if passing_yards_result:
            metrics = _categorize_metric(
                metrics,
                "✈️ Passing Yards Differential",
                passing_yards_result,
                reliability_data,
            )

    return metrics


def _categorize_metric(
    metrics,
    metric_name,
    result,
    reliability_data=None,
    min_sample_size=5,
    model_baseline=62.9,
):
    """
    Helper function to categorize a metric based on its accuracy relative to model baseline.
    Only includes the metric if the specific bucket has sufficient sample size.

    Categorization logic:
    - Positive: accuracy >= model_baseline (better than average)
    - Warning: model_baseline > accuracy >= (model_baseline - 5) (slightly below average)
    - Negative: accuracy < (model_baseline - 5) (significantly below average)

    Args:
        metrics: The metrics dict to append to
        metric_name: Display name for the metric
        result: The result dict from lookup_accuracy_metric (includes metric_type)
        reliability_data: Dict of reliability scores by metric_type
        min_sample_size: Minimum number of games required in this bucket (default: 5)
        model_baseline: Model's overall accuracy baseline (default: 62.9%)

    Returns:
        metrics dict (unchanged if sample size too small, but adds to stage2_filtered list)
    """
    # Check if this specific bucket has enough data
    if result["total"] < min_sample_size:
        # Skip this indicator - insufficient data for this specific bucket
        # Add to stage2_filtered list to show what was excluded for this game
        metrics["stage2_filtered"].append(
            {
                "name": metric_name,
                "category": result["category"],
                "sample_size": result["total"],
                "reason": f"Bucket has only {result['total']} games (need ≥{min_sample_size})",
            }
        )
        return metrics

    accuracy = result["accuracy"]
    metric_type = result.get("metric_type")  # Extract from result

    metric_info = {
        "name": metric_name,
        "category": result["category"],
        "accuracy": accuracy,
        "correct": result["correct"],
        "total": result["total"],
    }

    # Add reliability data if available
    if reliability_data and metric_type and metric_type in reliability_data:
        rel = reliability_data[metric_type]
        metric_info["reliability_score"] = round(rel["reliability_score"], 1)
        metric_info["positive_indicator_reliability"] = round(
            rel["positive_indicator_reliability"], 1
        )
        metric_info["positive_precision"] = round(rel["positive_precision"], 1)
        metric_info["positive_appearances"] = rel["positive_appearances"]
        metric_info["positive_in_correct"] = rel["positive_in_correct_predictions"]
        metric_info["positive_in_incorrect"] = rel["positive_in_incorrect_predictions"]
        metric_info["negative_indicator_reliability"] = round(
            rel["negative_indicator_reliability"], 1
        )
        metric_info["negative_precision"] = round(rel["negative_precision"], 1)
        metric_info["negative_appearances"] = rel["negative_appearances"]
        metric_info["negative_in_correct"] = rel["negative_in_correct_predictions"]
        metric_info["negative_in_incorrect"] = rel["negative_in_incorrect_predictions"]
    else:
        metric_info["reliability_score"] = None

    # Compare to model baseline instead of fixed thresholds
    if accuracy >= model_baseline:
        metrics["positive_indicators"].append(metric_info)
    elif accuracy >= (model_baseline - 5):
        metrics["warning_indicators"].append(metric_info)  # Renamed from neutral
    else:
        metrics["negative_indicators"].append(metric_info)

    return metrics


def calculate_overall_confidence_score(metrics, model_baseline=62.9):
    """
    Calculate an overall confidence score based on all indicators.
    Uses model baseline to determine relative confidence levels.

    Args:
        metrics: dict with positive_indicators, warning_indicators, negative_indicators
        model_baseline: float, the model's overall accuracy (default 62.9%)

    Returns:
        float: confidence score (0-100)
        str: confidence level (High/Warning/Negative)
    """
    positive_count = len(metrics["positive_indicators"])
    warning_count = len(metrics["warning_indicators"])  # Renamed from neutral
    negative_count = len(metrics["negative_indicators"])

    total_count = positive_count + warning_count + negative_count

    if total_count == 0:
        return 0, "Unknown"

    # Calculate weighted average
    positive_avg = (
        sum(m["accuracy"] for m in metrics["positive_indicators"])
        if positive_count > 0
        else 0
    )
    warning_avg = (
        sum(m["accuracy"] for m in metrics["warning_indicators"])
        if warning_count > 0
        else 0
    )
    negative_avg = (
        sum(m["accuracy"] for m in metrics["negative_indicators"])
        if negative_count > 0
        else 0
    )

    total_weight = positive_avg + warning_avg + negative_avg

    if total_count > 0:
        confidence_score = total_weight / total_count
    else:
        confidence_score = 0

    # Determine confidence level relative to model baseline
    if confidence_score >= model_baseline:
        confidence_level = "High"
    elif confidence_score >= (model_baseline - 5):
        confidence_level = "Warning"
    else:
        confidence_level = "Negative"

    # Calculate overall reliability score
    # This measures HOW RELIABLE the indicators are (not predicting outcome)
    # High reliability = indicators we can trust (whether positive or negative)
    # Low reliability = indicators that don't matter historically
    total_reliability = 0
    reliability_count = 0

    # Positive indicators: use their positive indicator reliability
    for metric in metrics["positive_indicators"]:
        if metric.get("positive_indicator_reliability"):
            reliability = metric["positive_indicator_reliability"]
            total_reliability += reliability
            reliability_count += 1

    # Warning indicators: use their negative indicator reliability (warnings are negative signals)
    for metric in metrics["warning_indicators"]:
        if metric.get("negative_indicator_reliability"):
            reliability = metric["negative_indicator_reliability"]
            total_reliability += reliability
            reliability_count += 1

    # Negative indicators: use their negative indicator reliability
    for metric in metrics["negative_indicators"]:
        if metric.get("negative_indicator_reliability"):
            reliability = metric["negative_indicator_reliability"]
            total_reliability += reliability
            reliability_count += 1

    # Calculate average reliability
    if reliability_count > 0:
        reliability_score = total_reliability / reliability_count
    else:
        reliability_score = 50.0  # Neutral if no reliability data

    metrics["reliability_score"] = round(reliability_score, 1)
    metrics["reliability_count"] = reliability_count

    return confidence_score, confidence_level


def detect_feature_conflicts(metrics):
    """
    Detect when top model features have conflicting historical performance.
    Returns list of conflicts found.
    """
    conflicts = []

    # Get top feature indicators
    top_features = [
        "⭐ Power Ranking L3 (Top Feature)",
        "Power Ranking Differential",
        "Avg Margin of Victory Diff",
        "Avg MOV Diff L3",
        "Avg Weekly Point Diff",
        "Passing EPA Differential",
        "Spread Performance Diff (ATS)",
    ]

    top_positive = [
        m
        for m in metrics["positive_indicators"]
        if any(feat in m["name"] for feat in top_features)
    ]
    top_negative = [
        m
        for m in metrics["negative_indicators"]
        if any(feat in m["name"] for feat in top_features)
    ]

    # Critical conflict: Top feature (Power Ranking L3) is negative
    pr_l3_negative = any("Power Ranking L3" in m["name"] for m in top_negative)
    if pr_l3_negative:
        pr_l3_metric = next(m for m in top_negative if "Power Ranking L3" in m["name"])
        conflicts.append(
            {
                "severity": "CRITICAL",
                "description": f"⚠️ HIGHEST IMPORTANCE FEATURE shows poor historical accuracy: {pr_l3_metric['category']} ({pr_l3_metric['accuracy']:.1f}%)",
                "recommendation": "Strong caution advised - the most important feature contradicts this pick",
            }
        )

    # Conflict: Multiple top features are negative
    if len(top_negative) >= 3:
        conflicts.append(
            {
                "severity": "HIGH",
                "description": f"⚠️ {len(top_negative)} high-importance features show poor historical accuracy",
                "recommendation": "Model may be overconfident - reduce stake significantly",
            }
        )

    # Warning: Mix of strong positive and negative top features
    if len(top_positive) >= 2 and len(top_negative) >= 2:
        conflicts.append(
            {
                "severity": "MEDIUM",
                "description": f"⚙️ Mixed signals: {len(top_positive)} positive vs {len(top_negative)} negative top features",
                "recommendation": "Model uncertainty - this could be a toss-up game",
            }
        )

    return conflicts


def get_confidence_class(score, model_baseline=62.9):
    """
    Return CSS class based on confidence score relative to model baseline.

    Args:
        score: overall confidence score
        model_baseline: model's overall accuracy (default 62.9%)
    """
    if score >= model_baseline:
        return "confidence-high"
    elif score >= (model_baseline - 5):
        return "confidence-warning"
    else:
        return "confidence-negative"


def get_accuracy_class(accuracy):
    """Return CSS class based on accuracy level."""
    if accuracy >= 62.9:
        return "accuracy-good"
    elif accuracy >= 57.9:
        return "accuracy-medium"
    else:
        return "accuracy-bad"


def get_filtered_metrics_info():
    """
    Get information about metric bucket filtering.
    Now shows per-bucket sample sizes and filtering status for both stages.

    Returns:
        dict with 'metrics' list containing bucket-level info,
        'excluded_metrics' list (failed Stage 1),
        'included_metrics' list (passed Stage 1)
    """
    # List of all available metrics to check
    all_metrics = [
        "spread_magnitude",
        "margin_magnitude",
        "power_ranking_diff",
        "adj_overall_rank_diff",
        "predicted_team",
        "pick_location",
        "favorite_underdog",
        "division_game",
        "primetime",
        "rest_differential",
        "qb_change",
        "qb_change_any",
        "close_game",
        "recent_form",
        "epa_mismatch",
        "weather",
        "power_ranking_diff_l3",
        "avg_margin_of_victory_diff",
        "avg_margin_of_victory_diff_l3",
        "avg_weekly_point_diff",
        "spread_performance_diff",
        "sacks_suffered_avg_diff",
        "completion_pct_diff",
        "yards_per_attempt_diff",
        "yards_per_carry_diff",
        "turnover_diff",
        "spread_diff",
        "rushing_yards_diff",
        "def_rushing_yards_diff",
        "passing_yards_diff",
    ]

    engine = get_db_engine()
    metrics_info = []
    excluded_metrics = []
    included_metrics = []

    for metric_type in all_metrics:
        # Get metric stats
        query = text(
            """
            SELECT category, accuracy, total
            FROM prediction_accuracy_metrics
            WHERE metric_type = :metric_type
        """
        )

        with engine.connect() as conn:
            result = conn.execute(query, {"metric_type": metric_type}).fetchall()

        if not result:
            continue

        accuracies = [row[1] for row in result]
        totals = [row[2] for row in result]

        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        variance = max_accuracy - min_accuracy
        min_sample = min(totals)
        max_sample = max(totals)

        # Count how many buckets meet the minimum sample size
        buckets_with_data = sum(1 for t in totals if t >= 5)
        total_buckets = len(totals)

        # Check if metric passes Stage 1 (is_metric_meaningful logic)
        usable_buckets = [
            (accuracies[i], totals[i]) for i in range(len(totals)) if totals[i] >= 5
        ]
        if len(usable_buckets) >= 2:
            usable_accuracies = [b[0] for b in usable_buckets]
            usable_variance = max(usable_accuracies) - min(usable_accuracies)
            is_meaningful = usable_variance >= 5.0
        else:
            usable_variance = 0
            is_meaningful = False

        metric_data = {
            "metric": metric_type,
            "variance": variance,
            "usable_variance": usable_variance,
            "min_sample": min_sample,
            "max_sample": max_sample,
            "buckets_with_data": buckets_with_data,
            "total_buckets": total_buckets,
            "avg_sample": sum(totals) / len(totals) if totals else 0,
            "is_meaningful": is_meaningful,
        }

        metrics_info.append(metric_data)

        if is_meaningful:
            included_metrics.append(metric_data)
        else:
            excluded_metrics.append(metric_data)

    return {
        "metrics": sorted(
            metrics_info, key=lambda x: x["usable_variance"], reverse=True
        ),
        "excluded_metrics": sorted(
            excluded_metrics, key=lambda x: x["usable_variance"], reverse=True
        ),
        "included_metrics": sorted(
            included_metrics, key=lambda x: x["usable_variance"], reverse=True
        ),
    }


def generate_future_predictions_report(output_file="nfl_future_prediction_report.html"):
    """Generate comprehensive HTML report for future predictions with outlier analysis."""

    # Load future predictions
    df = load_future_predictions()

    if len(df) == 0:
        print("⚠️  No future predictions found in database")
        return

    print(f"📊 Analyzing {len(df)} future predictions...")

    # Calculate metric reliability data once (used for all predictions)
    print("   📈 Calculating metric reliability scores...")
    reliability_data = calculate_metric_reliability()
    print(f"   ✓ Loaded reliability data for {len(reliability_data)} metrics")

    # Get filtered metrics info
    metrics_info = get_filtered_metrics_info()

    # Analyze each prediction
    predictions_with_metrics = []
    for _, prediction in df.iterrows():
        metrics = analyze_prediction_metrics(prediction, reliability_data)
        confidence_score, confidence_level = calculate_overall_confidence_score(metrics)
        conflicts = detect_feature_conflicts(metrics)

        predictions_with_metrics.append(
            {
                "prediction": prediction,
                "metrics": metrics,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "conflicts": conflicts,
            }
        )

    # Sort by confidence score (lowest to highest for outliers first)
    predictions_with_metrics.sort(key=lambda x: x["confidence_score"])

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NFL Future Predictions - Outlier Analysis</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #013369;
                border-bottom: 3px solid #D50A0A;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #013369;
                margin-top: 30px;
                border-bottom: 2px solid #ccc;
                padding-bottom: 5px;
            }}
            .prediction-card {{
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 5px solid #ccc;
            }}
            .prediction-card.high-confidence {{
                background-color: #d4edda;
                border-left-color: #28a745;
            }}
            .prediction-card.warning-confidence {{
                background-color: #fff3cd;
                border-left-color: #ffc107;
            }}
            .prediction-card.negative-confidence {{
                background-color: #f8d7da;
                border-left-color: #dc3545;
            }}
            .prediction-header {{
                display: grid;
                grid-template-columns: 2fr 1fr 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #eee;
            }}
            .matchup {{
                font-size: 24px;
                font-weight: bold;
                color: #013369;
            }}
            .game-info {{
                font-size: 14px;
                color: #666;
            }}
            .confidence-badge {{
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                text-align: center;
            }}
            .confidence-high {{
                background-color: #d4edda;
                color: #155724;
            }}
            .confidence-warning {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .confidence-negative {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .metrics-section {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .metric-box {{
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid;
            }}
            .metric-box.positive {{
                background-color: #d4edda;
                border-left-color: #28a745;
            }}
            .metric-box.warning {{
                background-color: #fff3cd;
                border-left-color: #ffc107;
            }}
            .metric-box.negative {{
                background-color: #f8d7da;
                border-left-color: #dc3545;
            }}
            .metric-title {{
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .metric-detail {{
                font-size: 13px;
                color: #555;
            }}
            .accuracy-good {{
                color: #28a745;
                font-weight: bold;
            }}
            .accuracy-medium {{
                color: #ffc107;
                font-weight: bold;
            }}
            .accuracy-bad {{
                color: #dc3545;
                font-weight: bold;
            }}
            .summary-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }}
            .summary-stat {{
                text-align: center;
            }}
            .summary-stat h3 {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .summary-stat .value {{
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .alert {{
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 4px solid;
            }}
            .alert.danger {{
                background-color: #f8d7da;
                border-left-color: #dc3545;
                color: #721c24;
            }}
            .alert.warning {{
                background-color: #fff3cd;
                border-left-color: #ffc107;
                color: #856404;
            }}
            .alert.info {{
                background-color: #d1ecf1;
                border-left-color: #17a2b8;
                color: #0c5460;
            }}
            .timestamp {{
                color: #666;
                font-size: 12px;
                text-align: right;
                margin-top: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th {{
                background-color: #013369;
                color: white;
                padding: 10px;
                text-align: left;
                font-size: 13px;
            }}
            td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <h1>🔮 NFL Future Predictions - Outlier Analysis</h1>
        
        <div class="summary-box">
            <h2 style="margin-top: 0; color: white;">📊 Analysis Summary</h2>
            <div class="summary-grid">
                <div class="summary-stat">
                    <h3>Total Predictions</h3>
                    <div class="value">{len(predictions_with_metrics)}</div>
                </div>
                <div class="summary-stat">
                    <h3>High Confidence (≥Baseline)</h3>
                    <div class="value">{sum(1 for p in predictions_with_metrics if p['confidence_level'] == 'High')}</div>
                </div>
                <div class="summary-stat">
                    <h3>Warning (Below Baseline)</h3>
                    <div class="value">{sum(1 for p in predictions_with_metrics if p['confidence_level'] == 'Warning')}</div>
                </div>
                <div class="summary-stat">
                    <h3>Negative (Well Below)</h3>
                    <div class="value">{sum(1 for p in predictions_with_metrics if p['confidence_level'] == 'Negative')}</div>
                </div>
            </div>
        </div>
        
        <div class="alert info">
            <strong>ℹ️ How to Read This Report:</strong><br>
            • <strong>Positive Indicators (Green):</strong> Historical accuracy ≥62.9% (model baseline) - These factors support the prediction<br>
            • <strong>Warning Indicators (Yellow):</strong> 57.9-62.9% accuracy (below baseline) - Proceed with caution<br>
            • <strong>Negative Indicators (Red):</strong> <57.9% accuracy (well below baseline) - Warning signs that may indicate an outlier<br>
            • <strong>Overall Confidence Score:</strong> Weighted average of all indicators, categorized relative to model baseline (62.9%):<br>
            &nbsp;&nbsp;- <strong>High:</strong> ≥62.9% (at or above model average)<br>
            &nbsp;&nbsp;- <strong>Warning:</strong> 57.9-62.9% (below model average)<br>
            &nbsp;&nbsp;- <strong>Negative:</strong> <57.9% (significantly below model average)<br>
            • <strong>Reliability Score:</strong> Average of all indicator reliability scores - measures how trustworthy the indicators are
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;">
            <h3 style="margin-top: 0; color: #013369;">🔧 Filter Controls</h3>
            <div style="display: flex; flex-direction: column; gap: 15px;">
                <div>
                    <label style="display: flex; align-items: center; font-size: 16px; cursor: pointer;">
                        <input type="checkbox" id="filterLowReliability" onchange="toggleLowReliabilityIndicators()" 
                               style="width: 20px; height: 20px; margin-right: 10px; cursor: pointer;">
                        <span>Hide indicators with low reliability</span>
                    </label>
                </div>
                <div style="display: flex; align-items: center; gap: 15px; margin-left: 30px;">
                    <label for="reliabilityThreshold" style="font-size: 14px; min-width: 150px;">
                        Reliability Threshold:
                    </label>
                    <input type="range" id="reliabilityThreshold" min="0" max="100" value="60" 
                           oninput="updateThresholdDisplay(); if(document.getElementById('filterLowReliability').checked) toggleLowReliabilityIndicators();"
                           style="flex: 1; max-width: 300px; cursor: pointer;">
                    <span id="thresholdValue" style="font-weight: bold; min-width: 50px; color: #013369;">60%</span>
                </div>
                <p style="margin: 5px 0 0 30px; font-size: 13px; color: #666;">
                    When enabled, only shows indicators where the directional reliability meets or exceeds your threshold. 
                    Adjust the slider to filter out metrics that don't consistently predict outcomes.
                </p>
            </div>
        </div>
        
        <script>
        function updateThresholdDisplay() {{
            const threshold = document.getElementById('reliabilityThreshold').value;
            document.getElementById('thresholdValue').textContent = threshold + '%';
        }}
        </script>
        
        <script>
        function toggleLowReliabilityIndicators() {{
            const filterEnabled = document.getElementById('filterLowReliability').checked;
            const threshold = parseFloat(document.getElementById('reliabilityThreshold').value);
            const allPredictionCards = document.querySelectorAll('.prediction-card');
            const modelBaseline = 62.9;
            
            // Track new confidence levels for summary update
            let newHighCount = 0;
            let newWarningCount = 0;
            let newNegativeCount = 0;
            
            allPredictionCards.forEach(card => {{
                const allMetricBoxes = card.querySelectorAll('.metric-box');
                let visibleMetrics = [];
                
                // First pass: determine which metrics to show/hide and collect visible ones
                allMetricBoxes.forEach(box => {{
                    const accuracy = parseFloat(box.getAttribute('data-accuracy'));
                    const reliability = parseFloat(box.getAttribute('data-reliability'));
                    
                    if (filterEnabled && reliability < threshold) {{
                        box.style.display = 'none';
                    }} else {{
                        box.style.display = 'block';
                        
                        if (accuracy && reliability) {{
                            visibleMetrics.push({{
                                accuracy: accuracy,
                                reliability: reliability,
                                isPositive: box.classList.contains('positive'),
                                isWarning: box.classList.contains('warning'),
                                isNegative: box.classList.contains('negative')
                            }});
                        }}
                    }}
                }});
                
                // Recalculate confidence score and reliability score
                let avgConfidence, avgReliability, confidenceLevel, confidenceClass, borderClass;
                
                if (filterEnabled && visibleMetrics.length > 0) {{
                    // Filter is ON and we have visible metrics - calculate new values
                    avgConfidence = visibleMetrics.reduce((sum, m) => sum + m.accuracy, 0) / visibleMetrics.length;
                    avgReliability = visibleMetrics.reduce((sum, m) => sum + m.reliability, 0) / visibleMetrics.length;
                }} else if (!filterEnabled) {{
                    // Filter is OFF - use original values
                    avgConfidence = parseFloat(card.getAttribute('data-original-confidence'));
                    avgReliability = parseFloat(card.getAttribute('data-original-reliability'));
                    confidenceLevel = card.getAttribute('data-original-level');
                    confidenceClass = 'confidence-' + confidenceLevel.toLowerCase();
                    borderClass = confidenceLevel.toLowerCase() + '-confidence';
                }} else {{
                    // Filter is ON but no visible metrics - use original values as fallback
                    avgConfidence = parseFloat(card.getAttribute('data-original-confidence'));
                    avgReliability = parseFloat(card.getAttribute('data-original-reliability'));
                }}
                
                // Determine new confidence level (only if not already set above when filter is OFF)
                if (!confidenceLevel) {{
                    if (avgConfidence >= modelBaseline) {{
                        confidenceLevel = 'High';
                        confidenceClass = 'confidence-high';
                        borderClass = 'high-confidence';
                    }} else if (avgConfidence >= (modelBaseline - 5)) {{
                        confidenceLevel = 'Warning';
                        confidenceClass = 'confidence-warning';
                        borderClass = 'warning-confidence';
                    }} else {{
                        confidenceLevel = 'Negative';
                        confidenceClass = 'confidence-negative';
                        borderClass = 'negative-confidence';
                    }}
                }}
                
                // Update the confidence badge and score using specific classes
                const confidenceLevelBadge = card.querySelector('.confidence-level-badge');
                if (confidenceLevelBadge) {{
                    confidenceLevelBadge.textContent = confidenceLevel;
                    confidenceLevelBadge.className = 'confidence-badge confidence-level-badge ' + confidenceClass;
                }}
                
                const confidenceScoreText = card.querySelector('.confidence-score-text');
                if (confidenceScoreText) {{
                    confidenceScoreText.textContent = avgConfidence.toFixed(1) + '% avg accuracy';
                }}
                
                // Update reliability badge
                const reliabilityBadge = card.querySelector('.reliability-badge');
                if (reliabilityBadge) {{
                    reliabilityBadge.textContent = avgReliability.toFixed(1) + '%';
                    
                    // Update reliability badge color
                    let reliabilityClass;
                    if (avgReliability >= modelBaseline) {{
                        reliabilityClass = 'confidence-high';
                    }} else if (avgReliability >= (modelBaseline - 5)) {{
                        reliabilityClass = 'confidence-warning';
                    }} else {{
                        reliabilityClass = 'confidence-negative';
                    }}
                    reliabilityBadge.className = 'confidence-badge reliability-badge ' + reliabilityClass;
                }}
                    
                // Update card border color based on new confidence level
                card.classList.remove('high-confidence', 'warning-confidence', 'negative-confidence');
                card.classList.add(borderClass);
                
                // Track for summary update
                if (confidenceLevel === 'High') {{
                    newHighCount++;
                }} else if (confidenceLevel === 'Warning') {{
                    newWarningCount++;
                }} else if (confidenceLevel === 'Negative') {{
                    newNegativeCount++;
                }}
                
                // Store new confidence level on card for later use
                card.setAttribute('data-new-confidence-level', confidenceLevel);
                card.setAttribute('data-new-confidence-score', avgConfidence.toFixed(1));
                card.setAttribute('data-new-reliability-score', avgReliability.toFixed(1));
                
                // Update indicator counts in section headers
                const posCount = visibleMetrics.filter(m => m.isPositive).length;
                const warnCount = visibleMetrics.filter(m => m.isWarning).length;
                const negCount = visibleMetrics.filter(m => m.isNegative).length;
                
                // Update h3 headers with new counts
                const headers = card.querySelectorAll('h3');
                headers.forEach(h3 => {{
                    const text = h3.textContent;
                    if (text.includes('Positive Indicators')) {{
                        h3.textContent = '✅ Positive Indicators (' + posCount + ')';
                    }} else if (text.includes('Warning Indicators')) {{
                        h3.textContent = '⚠️ Warning Indicators (' + warnCount + ')';
                    }} else if (text.includes('Negative Indicators')) {{
                        h3.textContent = '❌ Negative Indicators (' + negCount + ')';
                    }}
                }});
            }});
            
            // Update summary box counts at the top
            const summaryStats = document.querySelectorAll('.summary-stat .value');
            if (summaryStats.length >= 4) {{
                // summaryStats[0] is Total Predictions - don't change it
                summaryStats[1].textContent = newHighCount;  // High Confidence
                summaryStats[2].textContent = newWarningCount;  // Warning
                summaryStats[3].textContent = newNegativeCount;  // Negative
            }}
            
            // Reorder cards within each week section
            reorderCardsInWeeks();
            
            // Update summary tables at the bottom
            updateSummaryTables();
        }}
        
        function reorderCardsInWeeks() {{
            const filterEnabled = document.getElementById('filterLowReliability').checked;
            
            // Find all week sections
            const weekHeadings = Array.from(document.querySelectorAll('h2')).filter(h => h.textContent.includes('Week'));
            
            weekHeadings.forEach(heading => {{
                // Collect all cards in this week section
                const cards = [];
                let nextElement = heading.nextElementSibling;
                
                while (nextElement && nextElement.tagName !== 'H2') {{
                    if (nextElement.classList && nextElement.classList.contains('prediction-card')) {{
                        // Use new score if filter is on, original score if off
                        const score = filterEnabled 
                            ? parseFloat(nextElement.getAttribute('data-new-confidence-score'))
                            : parseFloat(nextElement.getAttribute('data-original-confidence'));
                        
                        if (score) {{
                            cards.push({{
                                element: nextElement,
                                score: score
                            }});
                        }}
                    }}
                    nextElement = nextElement.nextSibling;
                }}
                
                // Sort cards by confidence score (ascending - lowest confidence first to match original report style)
                cards.sort((a, b) => a.score - b.score);
                
                // Reinsert cards in sorted order
                let insertAfter = heading;
                cards.forEach(card => {{
                    insertAfter.parentNode.insertBefore(card.element, insertAfter.nextSibling);
                    insertAfter = card.element;
                }});
            }});
        }}
        
        function updateSummaryTables() {{
            const allPredictionCards = document.querySelectorAll('.prediction-card');
            
            // Collect predictions by their new confidence levels
            const highConfPreds = [];
            const warningConfPreds = [];
            const negativeConfPreds = [];
            
            allPredictionCards.forEach(card => {{
                const newLevel = card.getAttribute('data-new-confidence-level');
                const newScore = parseFloat(card.getAttribute('data-new-confidence-score'));
                const newReliability = parseFloat(card.getAttribute('data-new-reliability-score'));
                
                if (newLevel && newScore) {{
                    const pred = {{
                        card: card,
                        level: newLevel,
                        score: newScore,
                        reliability: newReliability
                    }};
                    
                    if (newLevel === 'High') {{
                        highConfPreds.push(pred);
                    }} else if (newLevel === 'Warning') {{
                        warningConfPreds.push(pred);
                    }} else if (newLevel === 'Negative') {{
                        negativeConfPreds.push(pred);
                    }}
                }}
            }});
            
            // Sort by confidence score (descending)
            highConfPreds.sort((a, b) => b.score - a.score);
            warningConfPreds.sort((a, b) => b.score - a.score);
            negativeConfPreds.sort((a, b) => b.score - a.score);
            
            // Update High Confidence table
            updateTable('high', highConfPreds);
            updateTable('warning', warningConfPreds);
            updateTable('negative', negativeConfPreds);
        }}
        
        function updateTable(type, predictions) {{
            // Find the table section by looking for the heading
            let heading;
            if (type === 'high') {{
                heading = Array.from(document.querySelectorAll('h2')).find(h => h.textContent.includes('High Confidence Predictions'));
            }} else if (type === 'warning') {{
                heading = Array.from(document.querySelectorAll('h2')).find(h => h.textContent.includes('Warning Confidence Predictions'));
            }} else if (type === 'negative') {{
                heading = Array.from(document.querySelectorAll('h2')).find(h => h.textContent.includes('Negative Confidence Predictions'));
            }}
            
            if (!heading) return;
            
            // Update heading count
            const originalText = heading.textContent.split('(')[0];
            heading.textContent = originalText + '(' + predictions.length + ' games)';
            
            // Find the table after this heading
            let table = heading.nextElementSibling;
            while (table && table.tagName !== 'TABLE') {{
                table = table.nextElementSibling;
            }}
            
            if (!table) return;
            
            // Clear existing rows (keep header)
            const tbody = table.querySelector('tbody') || table;
            const rows = Array.from(tbody.querySelectorAll('tr'));
            rows.forEach((row, idx) => {{
                if (idx > 0) row.remove(); // Keep header row
            }});
            
            // Add new rows
            predictions.forEach(pred => {{
                const card = pred.card;
                const predHeader = card.querySelector('.prediction-header');
                
                // Extract data from card
                const matchup = card.querySelector('.matchup').textContent.trim().replace(/⚠️.*$/, '').trim();
                const week = predHeader.textContent.match(/Week (\\d+)/)?.[1] || '';
                const winner = predHeader.textContent.match(/Predicted Winner\\s+([A-Z]+)/)?.[1] || '';
                const modelConf = predHeader.textContent.match(/Model Confidence\\s+(\\d+)/)?.[1] || '';
                
                // Count visible indicators
                const posCount = card.querySelectorAll('.metric-box.positive[style*="display: block"], .metric-box.positive:not([style*="display: none"])').length;
                const warnCount = card.querySelectorAll('.metric-box.warning[style*="display: block"], .metric-box.warning:not([style*="display: none"])').length;
                const negCount = card.querySelectorAll('.metric-box.negative[style*="display: block"], .metric-box.negative:not([style*="display: none"])').length;
                
                // Create new row
                const row = tbody.insertRow(-1);
                
                let accuracyClass = 'accuracy-good';
                if (pred.score < 62.9) accuracyClass = 'accuracy-medium';
                if (pred.score < 57.9) accuracyClass = 'accuracy-bad';
                
                let reliabilityClass = 'confidence-high';
                if (pred.reliability < 62.9) reliabilityClass = 'confidence-warning';
                if (pred.reliability < 57.9) reliabilityClass = 'confidence-negative';
                
                if (type === 'high') {{
                    row.innerHTML = `
                        <td>${{week}}</td>
                        <td>${{matchup}}</td>
                        <td><strong>${{winner}}</strong></td>
                        <td>${{modelConf}}</td>
                        <td class="${{accuracyClass}}">${{pred.score.toFixed(1)}}%</td>
                        <td class="${{reliabilityClass}}">${{pred.reliability.toFixed(1)}}%</td>
                        <td>${{posCount}} supporting factors</td>
                    `;
                }} else if (type === 'warning') {{
                    row.innerHTML = `
                        <td>${{week}}</td>
                        <td>${{matchup}}</td>
                        <td><strong>${{winner}}</strong></td>
                        <td>${{modelConf}}</td>
                        <td class="${{accuracyClass}}">${{pred.score.toFixed(1)}}%</td>
                        <td class="${{reliabilityClass}}">${{pred.reliability.toFixed(1)}}%</td>
                        <td>${{posCount}} positive</td>
                        <td>${{warnCount}} warning</td>
                    `;
                }} else if (type === 'negative') {{
                    row.innerHTML = `
                        <td>${{week}}</td>
                        <td>${{matchup}}</td>
                        <td><strong>${{winner}}</strong></td>
                        <td>${{modelConf}}</td>
                        <td class="${{accuracyClass}}">${{pred.score.toFixed(1)}}%</td>
                        <td class="${{reliabilityClass}}">${{pred.reliability.toFixed(1)}}%</td>
                        <td>${{warnCount}} warning</td>
                        <td>${{negCount}} negative</td>
                    `;
                }}
            }});
            
            // Hide section if no predictions
            if (predictions.length === 0) {{
                heading.style.display = 'none';
                if (heading.nextElementSibling && heading.nextElementSibling.classList.contains('alert')) {{
                    heading.nextElementSibling.style.display = 'none';
                }}
                if (table) table.style.display = 'none';
            }} else {{
                heading.style.display = 'block';
                if (heading.nextElementSibling && heading.nextElementSibling.classList.contains('alert')) {{
                    heading.nextElementSibling.style.display = 'block';
                }}
                if (table) table.style.display = 'table';
            }}
        }}
        </script>
        
        <h2>🔍 Metric Bucket Analysis</h2>
        <div class="alert info">
            <strong>📊 Two-Stage Filtering Approach:</strong><br>
            <strong>Stage 1 (Metric-Level):</strong> Metrics are first evaluated for meaningful variance. Buckets with <5 games are removed, then remaining buckets must show ≥5% variance. If variance is too low, the metric is excluded from ALL games.<br>
            <strong>Stage 2 (Bucket-Level):</strong> For metrics that pass Stage 1, individual buckets are checked per game. If a specific bucket has <5 games, it's excluded for that prediction only.<br>
            This two-stage approach prevents artificial variance from unreliable buckets while maximizing data usage.
        </div>
        
        <h3 style="color: #dc3545; margin-top: 20px;">❌ Stage 1: Metrics Excluded from ALL Games ({len(metrics_info['excluded_metrics'])}) metrics</h3>
        <div class="alert warning">
            These metrics failed Stage 1 filtering (insufficient variance after removing unreliable buckets). They are NOT used for any predictions.
        </div>
        <table>
            <tr>
                <th>Metric</th>
                <th>Raw Variance</th>
                <th>Usable Variance</th>
                <th>Usable Buckets</th>
                <th>Reason</th>
            </tr>
    """

    for metric in metrics_info["excluded_metrics"]:
        bucket_status = f"{metric['buckets_with_data']}/{metric['total_buckets']}"
        reason = (
            "Insufficient variance"
            if metric["usable_variance"] < 5
            else "Too few usable buckets"
        )

        html += f"""
            <tr>
                <td><strong>{metric['metric'].replace('_', ' ').title()}</strong></td>
                <td style="color: #666;">{metric['variance']:.1f}%</td>
                <td style="color: #dc3545; font-weight: bold;">{metric['usable_variance']:.1f}%</td>
                <td>{bucket_status}</td>
                <td style="color: #dc3545;">{reason}</td>
            </tr>
        """

    html += f"""
        </table>
        
        <h3 style="color: #28a745; margin-top: 20px;">✅ Stage 1: Metrics Available for Predictions ({len(metrics_info['included_metrics'])}) metrics</h3>
        <div class="alert info">
            These metrics passed Stage 1 filtering (≥5% variance in usable buckets). They are evaluated per game in Stage 2.
        </div>
        <table>
            <tr>
                <th>Metric</th>
                <th>Usable Variance</th>
                <th>Usable Buckets</th>
                <th>Sample Size Range</th>
                <th>Avg Sample</th>
            </tr>
    """

    for metric in metrics_info["included_metrics"]:
        bucket_status = f"{metric['buckets_with_data']}/{metric['total_buckets']}"
        bucket_class = (
            "accuracy-good"
            if metric["buckets_with_data"] == metric["total_buckets"]
            else "accuracy-medium"
        )

        html += f"""
            <tr>
                <td><strong>{metric['metric'].replace('_', ' ').title()}</strong></td>
                <td style="color: #28a745; font-weight: bold;">{metric['usable_variance']:.1f}%</td>
                <td class="{bucket_class}">{bucket_status}</td>
                <td>{metric['min_sample']}-{metric['max_sample']} games</td>
                <td>{metric['avg_sample']:.1f} games</td>
            </tr>
        """

    html += """
        </table>
        <div style="background-color: #e7f3ff; padding: 10px; margin-top: 10px; border-radius: 5px; font-size: 13px;">
            <strong>💡 How It Works:</strong><br>
            <strong>Stage 1:</strong> Remove buckets with <5 games, calculate variance on remaining buckets. If variance <5%, exclude metric from all predictions.<br>
            <strong>Stage 2:</strong> For metrics that passed Stage 1, check each game's specific bucket. Only include if that bucket has ≥5 games.<br>
            Example: A metric with buckets [60% (50 games), 61% (48 games), 62% (2 games)] → Remove 62% bucket → 1% variance → Excluded entirely (Stage 1).
        </div>
    """

    # Group predictions by week
    weeks = df["week"].unique()
    for week in sorted(weeks):
        week_predictions = [
            p for p in predictions_with_metrics if p["prediction"]["week"] == week
        ]

        html += f"""
        <h2>Week {week} Predictions ({len(week_predictions)} games)</h2>
        """

        for pred_data in week_predictions:
            prediction = pred_data["prediction"]
            metrics = pred_data["metrics"]
            confidence_score = pred_data["confidence_score"]
            confidence_level = pred_data["confidence_level"]
            conflicts = pred_data["conflicts"]

            # Determine if this is a potential outlier
            is_outlier = confidence_score < 50
            has_conflicts = len(conflicts) > 0
            outlier_badge = ""
            if is_outlier:
                outlier_badge = "<span style='background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; margin-left: 10px;'>⚠️ POTENTIAL OUTLIER</span>"
            elif has_conflicts:
                outlier_badge = "<span style='background-color: #ffc107; color: #333; padding: 5px 10px; border-radius: 15px; font-size: 12px; margin-left: 10px;'>⚙️ FEATURE CONFLICTS</span>"

            matchup = f"{prediction['away_team']} @ {prediction['home_team']}"

            html += f"""
            <div class="prediction-card {confidence_level.lower()}-confidence" 
                 data-week="{prediction['week']}" 
                 data-matchup="{matchup}"
                 data-winner="{prediction['predicted_winner']}"
                 data-model-confidence="{prediction['confidence']}"
                 data-original-confidence="{confidence_score:.1f}"
                 data-original-level="{confidence_level}"
                 data-original-reliability="{metrics['reliability_score']:.1f}">
                <div class="prediction-header">
                    <div>
                        <div class="matchup">{matchup}{outlier_badge}</div>
                        <div class="game-info">
                            Game ID: {prediction['game_id']}<br>
                            Spread: {prediction['spread_favorite']} {prediction['spread']}<br>
                            Division Game: {'Yes' if prediction['div_game'] == 1 else 'No'}
                        </div>
                    </div>
                    <div>
                        <strong>Predicted Winner</strong><br>
                        <span style="font-size: 20px; font-weight: bold; color: #013369;">{prediction['predicted_winner']}</span><br>
                        <span style="font-size: 14px; color: #666;">by {abs(prediction['cover_spread_by']):.1f} pts</span>
                    </div>
                    <div>
                        <strong>Model Confidence</strong><br>
                        <span style="font-size: 20px; font-weight: bold;">{prediction['confidence']}</span><br>
                        <span style="font-size: 12px; color: #666;">(Ranking)</span>
                    </div>
                    <div class="hist-confidence-container">
                        <strong>Historical Confidence</strong><br>
                        <span class="confidence-badge confidence-level-badge {get_confidence_class(confidence_score)}">{confidence_level}</span><br>
                        <span class="confidence-score-text" style="font-size: 14px; color: #666;">{confidence_score:.1f}% avg accuracy</span>
                    </div>
                    <div class="reliability-container">
                        <strong>Reliability Score</strong><br>
                        <span class="confidence-badge reliability-badge {get_confidence_class(metrics['reliability_score'])}">{metrics['reliability_score']:.1f}%</span><br>
                        <span style="font-size: 12px; color: #666;">How trustworthy the indicators are</span>
                    </div>
                </div>
            """

            # Show feature conflicts first (if any)
            if conflicts:
                for conflict in conflicts:
                    severity_color = {
                        "CRITICAL": "danger",
                        "HIGH": "danger",
                        "MEDIUM": "warning",
                    }.get(conflict["severity"], "warning")

                    html += f"""
                    <div class="alert {severity_color}">
                        <strong>{conflict['description']}</strong><br>
                        💡 {conflict['recommendation']}
                    </div>
                    """

            # Show warning based on confidence level relative to model baseline
            if confidence_level == "Negative":
                html += f"""
                <div class="alert danger">
                    <strong>🚫 HIGH RISK PREDICTION</strong><br>
                    Overall confidence is significantly below model baseline (62.9%). Multiple negative historical indicators present.
                    Consider skipping this game or significantly reducing stake.
                </div>
                """
            elif confidence_level == "Warning":
                html += f"""
                <div class="alert warning">
                    <strong>⚠️ CAUTION ADVISED</strong><br>
                    Overall confidence is below model baseline (62.9%). Some concerning historical indicators present.
                    Exercise caution and consider reducing stake.
                </div>
                """

            # Positive indicators
            if metrics["positive_indicators"]:
                html += f"""
                <h3 style="color: #28a745; margin-top: 20px;">✅ Positive Indicators ({len(metrics['positive_indicators'])})</h3>
                <div class="metrics-section">
                """
                for metric in sorted(
                    metrics["positive_indicators"],
                    key=lambda x: (
                        x.get("positive_indicator_reliability", 0)
                        if x.get("positive_indicator_reliability")
                        else x["accuracy"]
                    ),
                    reverse=True,
                ):
                    reliability_badge = ""
                    reliability_detail = ""
                    if metric.get("positive_indicator_reliability") is not None:
                        pos_rel = metric["positive_indicator_reliability"]
                        pos_prec = metric["positive_precision"]
                        pos_correct = metric["positive_in_correct"]
                        pos_incorrect = metric["positive_in_incorrect"]
                        pos_total = metric["positive_appearances"]
                        reliability_badge = f" <span style='background: #28a745; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>⭐ Positive Reliability: {pos_rel:.1f}%</span>"
                        reliability_detail = f"<br><strong>As Positive Indicator:</strong> {pos_prec:.1f}% precision ({pos_correct} correct, {pos_incorrect} incorrect in {pos_total} games)<br><strong>Reliability:</strong> {pos_rel:.1f}% (appears more often in correct predictions)"

                    pos_rel_value = metric.get("positive_indicator_reliability", 0) or 0
                    html += f"""
                    <div class="metric-box positive" data-accuracy="{metric['accuracy']:.1f}" data-reliability="{pos_rel_value:.1f}">
                        <div class="metric-title">{metric['name']}{reliability_badge}</div>
                        <div class="metric-detail">
                            Category: {metric['category']}<br>
                            Historical Accuracy: <span class="{get_accuracy_class(metric['accuracy'])}">{metric['accuracy']:.1f}%</span><br>
                            Sample Size: {metric['correct']}/{metric['total']} games{reliability_detail}
                        </div>
                    </div>
                    """
                html += "</div>"

            # Warning indicators
            if metrics["warning_indicators"]:
                html += f"""
                <h3 style="color: #ffc107; margin-top: 20px;">⚠️ Warning Indicators ({len(metrics['warning_indicators'])})</h3>
                <div class="metrics-section">
                """
                for metric in sorted(
                    metrics["warning_indicators"],
                    key=lambda x: (
                        x.get("positive_indicator_reliability", 0)
                        if x.get("positive_indicator_reliability")
                        else x["accuracy"]
                    ),
                    reverse=True,
                ):
                    reliability_badge = ""
                    reliability_detail = ""
                    if metric.get("negative_indicator_reliability") is not None:
                        neg_rel = metric["negative_indicator_reliability"]
                        neg_prec = metric["negative_precision"]
                        neg_correct = metric["negative_in_correct"]
                        neg_incorrect = metric["negative_in_incorrect"]
                        neg_total = metric["negative_appearances"]
                        reliability_badge = f" <span style='background: #ffc107; color: #333; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>⭐ Negative Reliability: {neg_rel:.1f}%</span>"
                        reliability_detail = f"<br><strong>As Negative Indicator:</strong> {neg_prec:.1f}% precision ({neg_correct} correct, {neg_incorrect} incorrect in {neg_total} games)<br><strong>Reliability:</strong> {neg_rel:.1f}% (appears more often in incorrect predictions)"

                    neg_rel_value = metric.get("negative_indicator_reliability", 0) or 0
                    html += f"""
                    <div class=\"metric-box warning\" data-accuracy=\"{metric['accuracy']:.1f}\" data-reliability=\"{neg_rel_value:.1f}\">
                        <div class=\"metric-title\">{metric['name']}{reliability_badge}</div>
                        <div class=\"metric-detail\">
                            Category: {metric['category']}<br>
                            Historical Accuracy: <span class=\"{get_accuracy_class(metric['accuracy'])}\">{metric['accuracy']:.1f}%</span><br>
                            Sample Size: {metric['correct']}/{metric['total']} games{reliability_detail}
                        </div>
                    </div>
                    """
                html += "</div>"

            # Negative indicators (red flags)
            if metrics["negative_indicators"]:
                html += f"""
                <h3 style="color: #dc3545; margin-top: 20px;">❌ Negative Indicators ({len(metrics['negative_indicators'])})</h3>
                <div class="metrics-section">
                """
                for metric in sorted(
                    metrics["negative_indicators"],
                    key=lambda x: (
                        x.get("negative_indicator_reliability", 0)
                        if x.get("negative_indicator_reliability")
                        else x["accuracy"]
                    ),
                    reverse=True,
                ):
                    reliability_badge = ""
                    reliability_detail = ""
                    if metric.get("negative_indicator_reliability") is not None:
                        neg_rel = metric["negative_indicator_reliability"]
                        neg_prec = metric["negative_precision"]
                        neg_correct = metric["negative_in_correct"]
                        neg_incorrect = metric["negative_in_incorrect"]
                        neg_total = metric["negative_appearances"]
                        reliability_badge = f" <span style='background: #dc3545; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>⭐ Negative Reliability: {neg_rel:.1f}%</span>"
                        reliability_detail = f"<br><strong>As Negative Indicator:</strong> {neg_prec:.1f}% precision ({neg_incorrect} incorrect, {neg_correct} correct in {neg_total} games)<br><strong>Reliability:</strong> {neg_rel:.1f}% (appears more often in incorrect predictions)"

                    neg_rel_value = metric.get("negative_indicator_reliability", 0) or 0
                    html += f"""
                    <div class="metric-box negative" data-accuracy="{metric['accuracy']:.1f}" data-reliability="{neg_rel_value:.1f}">
                        <div class="metric-title">{metric['name']}{reliability_badge}</div>
                        <div class="metric-detail">
                            Category: {metric['category']}<br>
                            Historical Accuracy: <span class="{get_accuracy_class(metric['accuracy'])}">{metric['accuracy']:.1f}%</span><br>
                            Sample Size: {metric['correct']}/{metric['total']} games{reliability_detail}<br>
                            <strong style="color: #dc3545;">⚠️ Model historically struggles in this scenario</strong>
                        </div>
                    </div>
                    """
                html += "</div>"

            # Stage 2 filtered metrics (per-game exclusions)
            if metrics["stage2_filtered"]:
                html += f"""
                <h3 style="color: #666; margin-top: 20px;">🚫 Stage 2: Metrics Excluded for THIS Game ({len(metrics['stage2_filtered'])} metrics)</h3>
                <div class="alert warning" style="background-color: #f8f9fa; border-left-color: #6c757d; color: #495057;">
                    <strong>These metrics passed Stage 1 (meaningful variance) but were excluded for this specific game due to insufficient bucket samples:</strong>
                </div>
                <table style="margin-top: 10px;">
                    <tr>
                        <th style="background-color: #6c757d;">Metric</th>
                        <th style="background-color: #6c757d;">Bucket Category</th>
                        <th style="background-color: #6c757d;">Sample Size</th>
                        <th style="background-color: #6c757d;">Reason</th>
                    </tr>
                """
                for filtered in metrics["stage2_filtered"]:
                    html += f"""
                    <tr>
                        <td><strong>{filtered['name']}</strong></td>
                        <td>{filtered['category']}</td>
                        <td style="color: #dc3545; font-weight: bold;">{filtered['sample_size']} games</td>
                        <td style="color: #dc3545; font-size: 12px;">{filtered['reason']}</td>
                    </tr>
                    """
                html += "</table>"

            # Top model features - highlighted section
            # Format new metrics values
            completion_pct_val = (
                f"{prediction['completion_pct_diff']:.1%}"
                if pd.notna(prediction.get("completion_pct_diff"))
                else "N/A"
            )
            ypa_val = (
                f"{prediction['yards_per_attempt_diff']:.2f}"
                if pd.notna(prediction.get("yards_per_attempt_diff"))
                else "N/A"
            )
            ypc_val = (
                f"{prediction['yards_per_carry_diff']:.2f}"
                if pd.notna(prediction.get("yards_per_carry_diff"))
                else "N/A"
            )
            turnover_val = (
                f"{prediction['turnover_diff']:+.0f}"
                if pd.notna(prediction.get("turnover_diff"))
                else "N/A"
            )
            rest_val = (
                f"{prediction['rest_diff']:+.0f}"
                if pd.notna(prediction.get("rest_diff"))
                else "N/A"
            )

            html += f"""
            <div style="margin-top: 20px; padding-top: 15px; border-top: 2px solid #013369;">
                <h4 style="margin: 0 0 10px 0; color: #013369;">⭐ Top 10 Model Features (by Importance)</h4>
                <table>
                    <tr>
                        <th style="width: 50%;">Feature</th>
                        <th style="width: 30%;">Value</th>
                        <th style="width: 20%;">Importance</th>
                    </tr>
                    <tr style="background-color: #fff3cd;">
                        <td><strong>1. Power Ranking Diff L3</strong></td>
                        <td><strong>{prediction.get('power_ranking_diff_l3', 'N/A') if pd.notna(prediction.get('power_ranking_diff_l3')) else 'N/A'}</strong></td>
                        <td><strong>9.06%</strong></td>
                    </tr>
                    <tr>
                        <td>2. Power Ranking Diff</td>
                        <td>{prediction['power_ranking_diff']:.3f}</td>
                        <td>7.29%</td>
                    </tr>
                    <tr>
                        <td>3. Avg Margin of Victory Diff</td>
                        <td>{prediction.get('avg_margin_of_victory_diff', 'N/A') if pd.notna(prediction.get('avg_margin_of_victory_diff')) else 'N/A'}</td>
                        <td>6.49%</td>
                    </tr>
                    <tr>
                        <td>4. Avg MOV Diff L3</td>
                        <td>{prediction.get('avg_margin_of_victory_diff_l3', 'N/A') if pd.notna(prediction.get('avg_margin_of_victory_diff_l3')) else 'N/A'}</td>
                        <td>4.84%</td>
                    </tr>
                    <tr>
                        <td>5. Avg Weekly Point Diff</td>
                        <td>{prediction.get('avg_weekly_point_diff', 'N/A') if pd.notna(prediction.get('avg_weekly_point_diff')) else 'N/A'}</td>
                        <td>4.47%</td>
                    </tr>
                    <tr>
                        <td>6. Passing EPA Diff</td>
                        <td>{prediction['passing_epa_diff']:.2f}</td>
                        <td>4.43%</td>
                    </tr>
                    <tr>
                        <td>7. Spread Performance Diff</td>
                        <td>{prediction.get('spread_performance_diff', 'N/A') if pd.notna(prediction.get('spread_performance_diff')) else 'N/A'}</td>
                        <td>4.11%</td>
                    </tr>
                    <tr>
                        <td>8. Avg Weekly Point Diff L3</td>
                        <td>{prediction['avg_weekly_point_diff_l3']:.2f}</td>
                        <td>4.01%</td>
                    </tr>
                    <tr>
                        <td>9. Rest Differential</td>
                        <td>{rest_val}</td>
                        <td>3.82%</td>
                    </tr>
                    <tr>
                        <td>10. Sacks Suffered Diff</td>
                        <td>{prediction.get('sacks_suffered_avg_diff', 'N/A') if pd.notna(prediction.get('sacks_suffered_avg_diff')) else 'N/A'}</td>
                        <td>3.54%</td>
                    </tr>
                </table>
                <div style="background-color: #e7f3ff; padding: 10px; margin-top: 10px; border-radius: 5px; font-size: 12px;">
                    <strong>ℹ️ Note:</strong> These features combine for ~51% of the model's decision-making. The top feature (Power Ranking L3) is the single most important factor.
                </div>
            </div>
            
            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd;">
                <h4 style="margin: 0 0 10px 0;">📊 Additional Game Metrics</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Adjusted Overall Rank Differential</td>
                        <td>{prediction['adj_overall_rank_diff']}</td>
                    </tr>
                    <tr>
                        <td>Win Percentage Differential</td>
                        <td>{prediction['win_pct_diff']:.3f}</td>
                    </tr>
                    <tr>
                        <td>Rushing EPA Differential</td>
                        <td>{prediction['rushing_epa_diff']:.2f}</td>
                    </tr>
                    <tr>
                        <td>QB Changes</td>
                        <td>Home: {prediction['home_qb_changed']}, Away: {prediction['away_qb_changed']}</td>
                    </tr>
                    <tr>
                        <td>Completion % Differential</td>
                        <td>{completion_pct_val}</td>
                    </tr>
                    <tr>
                        <td>Yards Per Attempt Differential</td>
                        <td>{ypa_val}</td>
                    </tr>
                    <tr>
                        <td>Yards Per Carry Differential</td>
                        <td>{ypc_val}</td>
                    </tr>
                    <tr>
                        <td>Turnover Differential</td>
                        <td>{turnover_val}</td>
                    </tr>
                    <tr>
                        <td>Rest Differential (days)</td>
                        <td>{rest_val}</td>
                    </tr>
                </table>
            </div>
            """

            html += "</div>"  # Close prediction-card

    # Summary of potential outliers
    outliers = [p for p in predictions_with_metrics if p["confidence_score"] < 50]
    if outliers:
        html += f"""
        <h2 style="color: #dc3545;">🚨 High Risk Predictions Summary ({len(outliers)} games)</h2>
        <div class="alert danger">
            <strong>These predictions have the highest risk of being incorrect based on historical patterns:</strong>
        </div>
        <table>
            <tr>
                <th>Week</th>
                <th>Matchup</th>
                <th>Predicted Winner</th>
                <th>Model Confidence</th>
                <th>Historical Confidence</th>
                <th>Avg Accuracy</th>
                <th>Negative Indicators</th>
            </tr>
        """
        for pred_data in outliers:
            prediction = pred_data["prediction"]
            html += f"""
            <tr>
                <td>{prediction['week']}</td>
                <td>{prediction['away_team']} @ {prediction['home_team']}</td>
                <td><strong>{prediction['predicted_winner']}</strong></td>
                <td>{prediction['confidence']}</td>
                <td><span class="confidence-badge {get_confidence_class(pred_data['confidence_score'])}">{pred_data['confidence_level']}</span></td>
                <td class="{get_accuracy_class(pred_data['confidence_score'])}">{pred_data['confidence_score']:.1f}%</td>
                <td>{len(pred_data['metrics']['negative_indicators'])} red flags</td>
            </tr>
            """
        html += "</table>"

    # High confidence picks (at or above model baseline)
    high_confidence = [
        p for p in predictions_with_metrics if p["confidence_level"] == "High"
    ]
    if high_confidence:
        html += f"""
        <h2 style="color: #28a745;">✅ High Confidence Predictions ({len(high_confidence)} games)</h2>
        <div class="alert info">
            <strong>These predictions meet or exceed model baseline accuracy (62.9%):</strong>
        </div>
        <table>
            <tr>
                <th>Week</th>
                <th>Matchup</th>
                <th>Predicted Winner</th>
                <th>Model Confidence</th>
                <th>Avg Accuracy</th>
                <th>Reliability Score</th>
                <th>Positive Indicators</th>
            </tr>
        """
        for pred_data in sorted(
            high_confidence, key=lambda x: x["confidence_score"], reverse=True
        ):
            prediction = pred_data["prediction"]
            reliability = pred_data["metrics"].get("reliability_score", 0)
            html += f"""
            <tr>
                <td>{prediction['week']}</td>
                <td>{prediction['away_team']} @ {prediction['home_team']}</td>
                <td><strong>{prediction['predicted_winner']}</strong></td>
                <td>{prediction['confidence']}</td>
                <td class="{get_accuracy_class(pred_data['confidence_score'])}">{pred_data['confidence_score']:.1f}%</td>
                <td class="{get_confidence_class(reliability)}">{reliability:.1f}%</td>
                <td>{len(pred_data['metrics']['positive_indicators'])} supporting factors</td>
            </tr>
            """
        html += "</table>"

    # Warning confidence picks (below baseline but not severely)
    warning_confidence = [
        p for p in predictions_with_metrics if p["confidence_level"] == "Warning"
    ]
    if warning_confidence:
        html += f"""
        <h2 style="color: #ffc107;">⚠️ Warning Confidence Predictions ({len(warning_confidence)} games)</h2>
        <div class="alert warning">
            <strong>These predictions are below model baseline (57.9-62.9% accuracy) - Exercise caution:</strong>
        </div>
        <table>
            <tr>
                <th>Week</th>
                <th>Matchup</th>
                <th>Predicted Winner</th>
                <th>Model Confidence</th>
                <th>Avg Accuracy</th>
                <th>Reliability Score</th>
                <th>Positive Indicators</th>
                <th>Warning Indicators</th>
            </tr>
        """
        for pred_data in sorted(
            warning_confidence, key=lambda x: x["confidence_score"], reverse=True
        ):
            prediction = pred_data["prediction"]
            reliability = pred_data["metrics"].get("reliability_score", 0)
            html += f"""
            <tr>
                <td>{prediction['week']}</td>
                <td>{prediction['away_team']} @ {prediction['home_team']}</td>
                <td><strong>{prediction['predicted_winner']}</strong></td>
                <td>{prediction['confidence']}</td>
                <td class="{get_accuracy_class(pred_data['confidence_score'])}">{pred_data['confidence_score']:.1f}%</td>
                <td class="{get_confidence_class(reliability)}">{reliability:.1f}%</td>
                <td>{len(pred_data['metrics']['positive_indicators'])} positive</td>
                <td>{len(pred_data['metrics']['warning_indicators'])} warning</td>
            </tr>
            """
        html += "</table>"

    # Negative confidence picks (significantly below baseline)
    negative_confidence = [
        p for p in predictions_with_metrics if p["confidence_level"] == "Negative"
    ]
    # Always generate section even if empty, so JavaScript can populate it
    html += f"""
    <h2 style="color: #dc3545;">🚫 Negative Confidence Predictions ({len(negative_confidence)} games)</h2>
    <div class="alert danger">
        <strong>These predictions are significantly below baseline (<57.9% accuracy) - High risk:</strong>
    </div>
    <table>
        <tr>
            <th>Week</th>
            <th>Matchup</th>
            <th>Predicted Winner</th>
            <th>Model Confidence</th>
            <th>Avg Accuracy</th>
            <th>Reliability Score</th>
            <th>Warning Indicators</th>
            <th>Negative Indicators</th>
        </tr>
    """
    for pred_data in sorted(
        negative_confidence, key=lambda x: x["confidence_score"], reverse=True
    ):
        prediction = pred_data["prediction"]
        reliability = pred_data["metrics"].get("reliability_score", 0)
        html += f"""
        <tr>
            <td>{prediction['week']}</td>
            <td>{prediction['away_team']} @ {prediction['home_team']}</td>
            <td><strong>{prediction['predicted_winner']}</strong></td>
            <td>{prediction['confidence']}</td>
            <td class="{get_accuracy_class(pred_data['confidence_score'])}">{pred_data['confidence_score']:.1f}%</td>
            <td class="{get_confidence_class(reliability)}">{reliability:.1f}%</td>
            <td>{len(pred_data['metrics']['warning_indicators'])} warning</td>
            <td>{len(pred_data['metrics']['negative_indicators'])} negative</td>
        </tr>
        """
    html += "</table>"

    # Hide section if initially empty
    if len(negative_confidence) == 0:
        html += """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const negativeHeading = Array.from(document.querySelectorAll('h2')).find(h => h.textContent.includes('Negative Confidence Predictions'));
            if (negativeHeading) {
                negativeHeading.style.display = 'none';
                if (negativeHeading.nextElementSibling && negativeHeading.nextElementSibling.classList.contains('alert')) {
                    negativeHeading.nextElementSibling.style.display = 'none';
                }
                let table = negativeHeading.nextElementSibling;
                while (table && table.tagName !== 'TABLE') {
                    table = table.nextElementSibling;
                }
                if (table) table.style.display = 'none';
            }
        });
        </script>
        """

    # --- Recalculate confidence scores for each threshold as the JS filter would ---
    thresholds = [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]

    matchups = [
        f"{p['prediction']['away_team']} @ {p['prediction']['home_team']}"
        for p in predictions_with_metrics
    ]
    matchup_set = sorted(set(matchups))
    # Build a mapping from matchup to predicted winner
    matchup_to_winner = {}
    for p in predictions_with_metrics:
        matchup = f"{p['prediction']['away_team']} @ {p['prediction']['home_team']}"
        winner = p["prediction"].get("predicted_winner", "")
        matchup_to_winner[matchup] = winner

    summary_data = {m: {} for m in matchup_set}
    for thresh in thresholds + ["nofilter"]:
        for p in predictions_with_metrics:
            matchup = f"{p['prediction']['away_team']} @ {p['prediction']['home_team']}"
            metrics = p["metrics"]
            filtered_metrics = []
            # Positive indicators: use positive_indicator_reliability
            for m in metrics["positive_indicators"]:
                rel = m.get("positive_indicator_reliability", 0)
                if thresh == "nofilter" or rel >= (
                    thresh if thresh != "nofilter" else 0
                ):
                    filtered_metrics.append(m)
            # Warning indicators: use negative_indicator_reliability (matches JS logic for warnings)
            for m in metrics["warning_indicators"]:
                rel = m.get("negative_indicator_reliability", 0)
                if thresh == "nofilter" or rel >= (
                    thresh if thresh != "nofilter" else 0
                ):
                    filtered_metrics.append(m)
            # Negative indicators: use negative_indicator_reliability
            for m in metrics["negative_indicators"]:
                rel = m.get("negative_indicator_reliability", 0)
                if thresh == "nofilter" or rel >= (
                    thresh if thresh != "nofilter" else 0
                ):
                    filtered_metrics.append(m)
            if filtered_metrics:
                avg_conf = sum(m["accuracy"] for m in filtered_metrics) / len(
                    filtered_metrics
                )
            else:
                avg_conf = 0
            summary_data[matchup][str(thresh)] = f"{avg_conf:.1f}%" if avg_conf else "-"

    # Build DataFrame for table
    summary_table_df = pd.DataFrame(
        [
            {
                "Matchup": m,
                "Predicted Winner": matchup_to_winner.get(m, ""),
                **summary_data[m],
            }
            for m in matchup_set
        ]
    )

    # Build HTML with colored cells using get_accuracy_class
    columns = summary_table_df.columns.tolist()
    summary_html = '<table class="accuracy-threshold-table">\n<tr>'
    for col in columns:
        summary_html += f"<th>{col}</th>"
    summary_html += "</tr>"
    for _, row in summary_table_df.iterrows():
        summary_html += "<tr>"
        for col in columns:
            val = row[col]
            if col == "Matchup" or val == "-":
                summary_html += f"<td>{val}</td>"
            else:
                try:
                    pct = float(str(val).replace("%", ""))
                    acc_class = get_accuracy_class(pct)
                except:
                    acc_class = ""
                summary_html += f'<td class="{acc_class}">{val}</td>'
        summary_html += "</tr>"
    summary_html += "</table>"

    html += "<h2>Accuracy by Reliability Threshold (Simulated Filter)</h2>"
    html += "<div style='margin-bottom: 20px;'>"
    html += summary_html
    html += "</div>"

    # --- Add ranking table ---
    # We'll use the same summary_table_df, but for each threshold column, rank the values (lowest=1)
    rank_df = summary_table_df.copy()
    rank_cols = [
        col for col in rank_df.columns if col not in ("Matchup", "Predicted Winner")
    ]

    for col in rank_cols:
        # Extract numeric values, set '-' as NaN
        vals = (
            rank_df[col].replace("-", float("nan")).str.replace("%", "").astype(float)
        )
        # Rank: lowest=1, highest=N, ties get min rank
        ranks = vals.rank(method="min")
        # Convert to int if not NaN, else blank
        rank_df[col] = ranks.apply(lambda x: str(int(x)) if not pd.isna(x) else "-")

    # Add average rank column
    avg_ranks = []
    for _, row in rank_df.iterrows():
        rank_vals = []
        for col in rank_cols:
            val = row[col]
            try:
                if val != "-":
                    rank_vals.append(int(val))
            except:
                pass
        avg_rank = sum(rank_vals) / len(rank_vals) if rank_vals else float("nan")
        avg_ranks.append(avg_rank)
    rank_df["Avg Rank"] = [f"{x:.2f}" if not pd.isna(x) else "-" for x in avg_ranks]

    # Sort by Avg Rank (highest first)
    def avg_rank_sort_key(row):
        try:
            return float(row["Avg Rank"]) if row["Avg Rank"] != "-" else float("-inf")
        except:
            return float("-inf")

    rank_df_sorted = rank_df.sort_values(
        by="Avg Rank",
        key=lambda col: col.replace("-", str(float("-inf"))).astype(float),
        ascending=False,
    )

    # Build HTML for ranking table with new color logic and Avg Rank column
    rank_html = '<table class="accuracy-threshold-table">\n<tr>'
    for col in rank_df_sorted.columns:
        rank_html += f"<th>{col}</th>"
    rank_html += "</tr>"
    n_rows = len(rank_df_sorted)
    for i, row in rank_df_sorted.iterrows():
        rank_html += "<tr>"
        for col in rank_df_sorted.columns:
            val = row[col]
            cell_class = ""
            if col == "Avg Rank" and val != "-":
                # Color avg rank: red for <=5, yellow for <=10, green for >10
                try:
                    avg_val = float(val)
                    if avg_val <= 5:
                        cell_class = "accuracy-bad"
                    elif avg_val <= 10:
                        cell_class = "accuracy-medium"
                    else:
                        cell_class = "accuracy-good"
                except:
                    pass
            else:
                try:
                    if (
                        col not in ("Matchup", "Predicted Winner", "Avg Rank")
                        and val != "-"
                    ):
                        rank_val = int(val)
                        if 1 <= rank_val <= 5:
                            cell_class = "accuracy-bad"  # Red for 1-5
                        elif 6 <= rank_val <= 10:
                            cell_class = "accuracy-medium"  # Yellow for 6-10
                        elif 11 <= rank_val <= 16:
                            cell_class = "accuracy-good"  # Green for 11-16
                except:
                    pass
            if cell_class:
                rank_html += f'<td class="{cell_class}">{val}</td>'
            else:
                rank_html += f"<td>{val}</td>"
        rank_html += "</tr>"
    rank_html += "</table>"

    html += "<h2>Accuracy Rank by Reliability Threshold (1 = Lowest Avg Accuracy)</h2>"
    html += "<div style='margin-bottom: 20px;'>"
    html += rank_html
    html += "</div>"

    html += f"""
        <div class="timestamp">
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Data source: future_predictions table
        </div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Future outlier report generated: {output_file}")
    print(f"   Total predictions analyzed: {len(predictions_with_metrics)}")
    print(f"   High confidence picks (≥baseline): {len(high_confidence)}")
    print(f"   Warning confidence picks (below baseline): {len(warning_confidence)}")
    print(f"   Negative confidence picks (well below): {len(negative_confidence)}")
    print(f"   Potential outliers (high risk): {len(outliers)}")


if __name__ == "__main__":
    generate_future_predictions_report()
