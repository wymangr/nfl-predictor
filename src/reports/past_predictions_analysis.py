"""Bucket analysis functions for analyzing prediction performance by metric ranges."""

import pandas as pd
from src.helpers.database_helpers import run_query


def get_bucket_analysis(table_name="past_predictions"):
    """
    Analyze prediction accuracy across all metric buckets.

    Args:
        table_name: Table to analyze ('past_predictions' or 'future_predictions')

    Returns:
        DataFrame with columns: metric, bucket, total, correct, accuracy_pct, vs_baseline, bucket_min, bucket_max
    """
    # Get baseline accuracy
    baseline_result = run_query(
        f"""
    SELECT 
        COUNT(*) AS total_predictions,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name} 
    WHERE correct != 'push'
    """
    )
    baseline_acc = baseline_result[0]["accuracy_pct"]

    # Collect all bucket analyses
    all_buckets = []

    # Confidence Score
    buckets = run_query(
        f"""
    SELECT 
        'Confidence Score' AS metric,
        CASE 
            WHEN confidence_score >= 95 THEN '95-100'
            WHEN confidence_score >= 90 THEN '90-94.99'
            WHEN confidence_score >= 85 THEN '85-89.99'
            WHEN confidence_score >= 80 THEN '80-84.99'
            WHEN confidence_score >= 75 THEN '75-79.99'
            ELSE 'Below 75'
        END AS bucket,
        MIN(confidence_score) as bucket_min,
        MAX(confidence_score) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Confidence Level
    buckets = run_query(
        f"""
    SELECT 
        'Confidence Level' AS metric,
        CASE 
            WHEN confidence >= 13 THEN '13-16 (Highest)'
            WHEN confidence >= 9 THEN '9-12 (High)'
            WHEN confidence >= 5 THEN '5-8 (Medium)'
            ELSE '1-4 (Low)'
        END AS bucket,
        MIN(confidence) as bucket_min,
        MAX(confidence) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Division Games
    buckets = run_query(
        f"""
    SELECT 
        'Division Game' AS metric,
        CASE WHEN div_game = 1 THEN 'Yes' ELSE 'No' END AS bucket,
        CAST(div_game AS FLOAT) as bucket_min,
        CAST(div_game AS FLOAT) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket, div_game
    """
    )
    all_buckets.extend(buckets)

    # Spread
    buckets = run_query(
        f"""
    SELECT 
        'Spread' AS metric,
        CASE 
            WHEN spread >= 10 THEN '10+ (Huge favorite)'
            WHEN spread >= 7 THEN '7-9.5'
            WHEN spread >= 4 THEN '4-6.5'
            WHEN spread >= 2 THEN '2-3.5'
            ELSE '<2 (Close game)'
        END AS bucket,
        MIN(spread) as bucket_min,
        MAX(spread) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Power Ranking Differential
    buckets = run_query(
        f"""
    SELECT 
        'Power Ranking Diff' AS metric,
        CASE 
            WHEN power_ranking_diff >= 0.3 THEN '0.3+ (Big home edge)'
            WHEN power_ranking_diff >= 0.15 THEN '0.15-0.299'
            WHEN power_ranking_diff >= 0 THEN '0-0.149 (Slight home edge)'
            WHEN power_ranking_diff >= -0.15 THEN '-0.001 to -0.149 (Slight away edge)'
            WHEN power_ranking_diff >= -0.3 THEN '-0.15 to -0.299'
            ELSE '<-0.3 (Big away edge)'
        END AS bucket,
        MIN(power_ranking_diff) as bucket_min,
        MAX(power_ranking_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # QB Changes
    buckets = run_query(
        f"""
    SELECT 
        'QB Changes' AS metric,
        CASE 
            WHEN home_qb_changed = 'Y' AND away_qb_changed = 'Y' THEN 'Both QBs Changed'
            WHEN home_qb_changed = 'Y' THEN 'Home QB Changed'
            WHEN away_qb_changed = 'Y' THEN 'Away QB Changed'
            ELSE 'No QB Changes'
        END AS bucket,
        NULL as bucket_min,
        NULL as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Predicted Margin
    buckets = run_query(
        f"""
    SELECT 
        'Predicted Margin' AS metric,
        CASE 
            WHEN ABS(predicted_diff) >= 14 THEN '14+ (Blowout)'
            WHEN ABS(predicted_diff) >= 10 THEN '10-13.99'
            WHEN ABS(predicted_diff) >= 7 THEN '7-9.99'
            WHEN ABS(predicted_diff) >= 4 THEN '4-6.99'
            ELSE '<4 (Close)'
        END AS bucket,
        MIN(ABS(predicted_diff)) as bucket_min,
        MAX(ABS(predicted_diff)) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Recent Form
    buckets = run_query(
        f"""
    SELECT 
        'Recent Form' AS metric,
        CASE 
            WHEN avg_weekly_point_diff_l3 >= 10 THEN '10+ (Dominant)'
            WHEN avg_weekly_point_diff_l3 >= 5 THEN '5-9.99'
            WHEN avg_weekly_point_diff_l3 >= 0 THEN '0-4.99'
            WHEN avg_weekly_point_diff_l3 >= -5 THEN '-0.01 to -4.99'
            WHEN avg_weekly_point_diff_l3 >= -10 THEN '-5 to -9.99'
            ELSE '<-10 (Poor form)'
        END AS bucket,
        MIN(avg_weekly_point_diff_l3) as bucket_min,
        MAX(avg_weekly_point_diff_l3) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push' AND avg_weekly_point_diff_l3 IS NOT NULL
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Rushing EPA
    buckets = run_query(
        f"""
    SELECT 
        'Rushing EPA' AS metric,
        CASE 
            WHEN rushing_epa_diff >= 30 THEN '30+ (Huge advantage)'
            WHEN rushing_epa_diff >= 15 THEN '15-29.99'
            WHEN rushing_epa_diff >= 0 THEN '0-14.99 (Advantage)'
            WHEN rushing_epa_diff >= -15 THEN '-0.01 to -14.99 (Disadvantage)'
            WHEN rushing_epa_diff >= -30 THEN '-15 to -29.99'
            ELSE '<-30 (Huge disadvantage)'
        END AS bucket,
        MIN(rushing_epa_diff) as bucket_min,
        MAX(rushing_epa_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push' AND rushing_epa_diff IS NOT NULL
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Passing EPA
    buckets = run_query(
        f"""
    SELECT 
        'Passing EPA' AS metric,
        CASE 
            WHEN passing_epa_diff >= 150 THEN '150+ (Huge advantage)'
            WHEN passing_epa_diff >= 75 THEN '75-149.99'
            WHEN passing_epa_diff >= 0 THEN '0-74.99 (Advantage)'
            WHEN passing_epa_diff >= -75 THEN '-0.01 to -74.99 (Disadvantage)'
            WHEN passing_epa_diff >= -150 THEN '-75 to -149.99'
            ELSE '<-150 (Huge disadvantage)'
        END AS bucket,
        MIN(passing_epa_diff) as bucket_min,
        MAX(passing_epa_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push' AND passing_epa_diff IS NOT NULL
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # YPA Differential
    buckets = run_query(
        f"""
    SELECT 
        'YPA Differential' AS metric,
        CASE 
            WHEN yards_per_attempt_diff >= 1.5 THEN '1.5+ (Big advantage)'
            WHEN yards_per_attempt_diff >= 0.5 THEN '0.5-1.49'
            WHEN yards_per_attempt_diff >= -0.5 THEN '-0.49 to 0.49 (Neutral)'
            WHEN yards_per_attempt_diff >= -1.5 THEN '-0.5 to -1.49'
            ELSE '<-1.5 (Big disadvantage)'
        END AS bucket,
        MIN(yards_per_attempt_diff) as bucket_min,
        MAX(yards_per_attempt_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # YPC Differential
    buckets = run_query(
        f"""
    SELECT 
        'YPC Differential' AS metric,
        CASE 
            WHEN yards_per_carry_diff >= 0.75 THEN '0.75+ (Big advantage)'
            WHEN yards_per_carry_diff >= 0.25 THEN '0.25-0.749'
            WHEN yards_per_carry_diff >= -0.25 THEN '-0.249 to 0.249 (Neutral)'
            WHEN yards_per_carry_diff >= -0.75 THEN '-0.25 to -0.749'
            ELSE '<-0.75 (Big disadvantage)'
        END AS bucket,
        MIN(yards_per_carry_diff) as bucket_min,
        MAX(yards_per_carry_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Win Pct Differential
    buckets = run_query(
        f"""
    SELECT 
        'Win Pct Diff' AS metric,
        CASE 
            WHEN win_pct_diff >= 0.3 THEN '0.3+ (30%+ better)'
            WHEN win_pct_diff >= 0.15 THEN '0.15-0.299'
            WHEN win_pct_diff >= 0 THEN '0-0.149 (Slight edge)'
            WHEN win_pct_diff >= -0.15 THEN '-0.001 to -0.149'
            ELSE '<-0.15 (Underdog pick)'
        END AS bucket,
        MIN(win_pct_diff) as bucket_min,
        MAX(win_pct_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Rank Differential
    buckets = run_query(
        f"""
    SELECT 
        'Rank Differential' AS metric,
        CASE 
            WHEN adj_overall_rank_diff >= 15 THEN '15+ (Much worse team picked)'
            WHEN adj_overall_rank_diff >= 7 THEN '7-14'
            WHEN adj_overall_rank_diff >= 0 THEN '0-6 (Slightly worse picked)'
            WHEN adj_overall_rank_diff >= -7 THEN '-1 to -6 (Better team picked)'
            ELSE '<-7 (Much better team picked)'
        END AS bucket,
        MIN(adj_overall_rank_diff) as bucket_min,
        MAX(adj_overall_rank_diff) as bucket_max,
        COUNT(*) AS total,
        SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) AS correct,
        ROUND(100.0 * SUM(CASE WHEN correct = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) AS accuracy_pct
    FROM {table_name}
    WHERE correct != 'push'
    GROUP BY bucket
    """
    )
    all_buckets.extend(buckets)

    # Convert to DataFrame
    df = pd.DataFrame(all_buckets)
    df["vs_baseline"] = df["accuracy_pct"] - baseline_acc
    df["baseline_accuracy"] = baseline_acc

    return df


def classify_prediction_bucket(value, metric_name):
    """
    Classify a prediction value into its appropriate bucket.

    Args:
        value: The metric value
        metric_name: Name of the metric

    Returns:
        Bucket name as string
    """
    if value is None:
        return None

    if metric_name == "Confidence Score":
        if value >= 95:
            return "95-100"
        elif value >= 90:
            return "90-94.99"
        elif value >= 85:
            return "85-89.99"
        elif value >= 80:
            return "80-84.99"
        elif value >= 75:
            return "75-79.99"
        else:
            return "Below 75"

    elif metric_name == "Confidence Level":
        if value >= 13:
            return "13-16 (Highest)"
        elif value >= 9:
            return "9-12 (High)"
        elif value >= 5:
            return "5-8 (Medium)"
        else:
            return "1-4 (Low)"

    elif metric_name == "Division Game":
        return "Yes" if value == 1 else "No"

    elif metric_name == "Spread":
        if value >= 10:
            return "10+ (Huge favorite)"
        elif value >= 7:
            return "7-9.5"
        elif value >= 4:
            return "4-6.5"
        elif value >= 2:
            return "2-3.5"
        else:
            return "<2 (Close game)"

    elif metric_name == "Power Ranking Diff":
        if value >= 0.3:
            return "0.3+ (Big home edge)"
        elif value >= 0.15:
            return "0.15-0.299"
        elif value >= 0:
            return "0-0.149 (Slight home edge)"
        elif value >= -0.15:
            return "-0.001 to -0.149 (Slight away edge)"
        elif value >= -0.3:
            return "-0.15 to -0.299"
        else:
            return "<-0.3 (Big away edge)"

    elif metric_name == "Predicted Margin":
        abs_value = abs(value)
        if abs_value >= 14:
            return "14+ (Blowout)"
        elif abs_value >= 10:
            return "10-13.99"
        elif abs_value >= 7:
            return "7-9.99"
        elif abs_value >= 4:
            return "4-6.99"
        else:
            return "<4 (Close)"

    elif metric_name == "Recent Form":
        if value >= 10:
            return "10+ (Dominant)"
        elif value >= 5:
            return "5-9.99"
        elif value >= 0:
            return "0-4.99"
        elif value >= -5:
            return "-0.01 to -4.99"
        elif value >= -10:
            return "-5 to -9.99"
        else:
            return "<-10 (Poor form)"

    elif metric_name == "Rushing EPA":
        if value >= 30:
            return "30+ (Huge advantage)"
        elif value >= 15:
            return "15-29.99"
        elif value >= 0:
            return "0-14.99 (Advantage)"
        elif value >= -15:
            return "-0.01 to -14.99 (Disadvantage)"
        elif value >= -30:
            return "-15 to -29.99"
        else:
            return "<-30 (Huge disadvantage)"

    elif metric_name == "Passing EPA":
        if value >= 150:
            return "150+ (Huge advantage)"
        elif value >= 75:
            return "75-149.99"
        elif value >= 0:
            return "0-74.99 (Advantage)"
        elif value >= -75:
            return "-0.01 to -74.99 (Disadvantage)"
        elif value >= -150:
            return "-75 to -149.99"
        else:
            return "<-150 (Huge disadvantage)"

    elif metric_name == "YPA Differential":
        if value >= 1.5:
            return "1.5+ (Big advantage)"
        elif value >= 0.5:
            return "0.5-1.49"
        elif value >= -0.5:
            return "-0.49 to 0.49 (Neutral)"
        elif value >= -1.5:
            return "-0.5 to -1.49"
        else:
            return "<-1.5 (Big disadvantage)"

    elif metric_name == "YPC Differential":
        if value >= 0.75:
            return "0.75+ (Big advantage)"
        elif value >= 0.25:
            return "0.25-0.749"
        elif value >= -0.25:
            return "-0.249 to 0.249 (Neutral)"
        elif value >= -0.75:
            return "-0.25 to -0.749"
        else:
            return "<-0.75 (Big disadvantage)"

    elif metric_name == "Win Pct Diff":
        if value >= 0.3:
            return "0.3+ (30%+ better)"
        elif value >= 0.15:
            return "0.15-0.299"
        elif value >= 0:
            return "0-0.149 (Slight edge)"
        elif value >= -0.15:
            return "-0.001 to -0.149"
        else:
            return "<-0.15 (Underdog pick)"

    elif metric_name == "Rank Differential":
        if value >= 15:
            return "15+ (Much worse team picked)"
        elif value >= 7:
            return "7-14"
        elif value >= 0:
            return "0-6 (Slightly worse picked)"
        elif value >= -7:
            return "-1 to -6 (Better team picked)"
        else:
            return "<-7 (Much better team picked)"

    return None


def calculate_bucket_confidence(prediction_dict, bucket_df):
    """
    Calculate bucket-based confidence for a single prediction.

    This is the shared function used by predict.py, analyze_future_bucket_confidence.py,
    and analyze_past_bucket_confidence.py to eliminate code duplication.

    Args:
        prediction_dict: Dictionary or pandas Series with prediction metrics
        bucket_df: DataFrame with bucket analysis data from get_bucket_analysis()

    Returns:
        tuple: (bucket_confidence, bucket_details)
            - bucket_confidence: weighted average accuracy across all buckets (float or None)
            - bucket_details: list of dicts with metric, bucket, accuracy, sample_size info
    """
    import math
    import pandas as pd

    # Metric mapping
    metrics_to_analyze = [
        ("confidence_score", "Confidence Score"),
        ("confidence", "Confidence Level"),
        ("div_game", "Division Game"),
        ("spread", "Spread"),
        ("power_ranking_diff", "Power Ranking Diff"),
        ("predicted_diff", "Predicted Margin"),
        ("avg_weekly_point_diff_l3", "Recent Form"),
        ("rushing_epa_diff", "Rushing EPA"),
        ("passing_epa_diff", "Passing EPA"),
        ("yards_per_attempt_diff", "YPA Differential"),
        ("yards_per_carry_diff", "YPC Differential"),
        ("win_pct_diff", "Win Pct Diff"),
        ("adj_overall_rank_diff", "Rank Differential"),
    ]

    bucket_accuracies = []
    bucket_details = []

    # Analyze each metric
    for col_name, metric_name in metrics_to_analyze:
        value = prediction_dict.get(col_name)
        if (
            value is None
            or (isinstance(value, float) and math.isnan(value))
            or (pd.isna(value))
        ):
            continue

        bucket = classify_prediction_bucket(value, metric_name)
        if bucket is None:
            continue

        # Look up historical performance for this bucket
        bucket_perf = bucket_df[
            (bucket_df["metric"] == metric_name) & (bucket_df["bucket"] == bucket)
        ]

        if len(bucket_perf) > 0:
            acc = bucket_perf.iloc[0]["accuracy_pct"]
            vs_base = bucket_perf.iloc[0]["vs_baseline"]
            sample_size = bucket_perf.iloc[0]["total"]

            bucket_accuracies.append(acc)
            bucket_details.append(
                {
                    "metric": metric_name,
                    "bucket": bucket,
                    "historical_accuracy": acc,
                    "vs_baseline": vs_base,
                    "sample_size": sample_size,
                }
            )

    # Handle QB changes
    home_qb_changed = prediction_dict.get("home_qb_changed", "N")
    away_qb_changed = prediction_dict.get("away_qb_changed", "N")

    if home_qb_changed == "Y" and away_qb_changed == "Y":
        qb_bucket = "Both QBs Changed"
    elif home_qb_changed == "Y":
        qb_bucket = "Home QB Changed"
    elif away_qb_changed == "Y":
        qb_bucket = "Away QB Changed"
    else:
        qb_bucket = "No QB Changes"

    qb_perf = bucket_df[
        (bucket_df["metric"] == "QB Changes") & (bucket_df["bucket"] == qb_bucket)
    ]
    if len(qb_perf) > 0:
        acc = qb_perf.iloc[0]["accuracy_pct"]
        vs_base = qb_perf.iloc[0]["vs_baseline"]
        sample_size = qb_perf.iloc[0]["total"]

        bucket_accuracies.append(acc)
        bucket_details.append(
            {
                "metric": "QB Changes",
                "bucket": qb_bucket,
                "historical_accuracy": acc,
                "vs_baseline": vs_base,
                "sample_size": sample_size,
            }
        )

    # Calculate weighted average bucket confidence
    if not bucket_accuracies:
        return None, []

    total_weight = sum([bd["sample_size"] for bd in bucket_details])
    weighted_acc = (
        sum([bd["historical_accuracy"] * bd["sample_size"] for bd in bucket_details])
        / total_weight
    )

    return weighted_acc, bucket_details


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_bucket_analysis():
    """Print detailed bucket analysis with visual formatting."""
    # Get bucket data
    print("Loading bucket analysis...")
    df = get_bucket_analysis("past_predictions")

    baseline_acc = df["baseline_accuracy"].iloc[0]

    # Overall baseline
    print_section("OVERALL BASELINE")
    print(f"Baseline Accuracy: {baseline_acc}%")
    print(f"Total Buckets Analyzed: {len(df)}")

    # Group by metric and display
    metrics = df["metric"].unique()

    for metric in metrics:
        print_section(metric.upper())
        metric_df = df[df["metric"] == metric].sort_values(
            "bucket_min", ascending=False, na_position="first"
        )

        for _, row in metric_df.iterrows():
            diff = row["vs_baseline"]
            marker = "⚠️ " if diff < -5 else "✓" if diff > 5 else "  "
            bucket_display = (
                f"{row['bucket']:40s}"
                if len(row["bucket"]) > 30
                else f"{row['bucket']:30s}"
            )
            print(
                f"{marker} {bucket_display} | N={row['total']:3d} | Acc={row['accuracy_pct']:5.2f}% ({diff:+.2f})"
            )

    # Top 20 worst performing buckets
    print_section("TOP 20 WORST PERFORMING BUCKETS (Cross-Metric)")
    worst = df[df["vs_baseline"] < 0].nsmallest(20, "accuracy_pct")
    for _, row in worst.iterrows():
        print(
            f"  {row['metric']:20s} | {row['bucket']:35s} | N={row['total']:3d} | Acc={row['accuracy_pct']:5.2f}% ({row['vs_baseline']:+.2f})"
        )

    # Top 20 best performing buckets
    print_section("TOP 20 BEST PERFORMING BUCKETS (Cross-Metric)")
    best = df[df["vs_baseline"] > 0].nlargest(20, "accuracy_pct")
    for _, row in best.iterrows():
        print(
            f"  {row['metric']:20s} | {row['bucket']:35s} | N={row['total']:3d} | Acc={row['accuracy_pct']:5.2f}% ({row['vs_baseline']:+.2f})"
        )

    # Summary statistics
    print_section("KEY FINDINGS SUMMARY")
    print(f"Baseline Accuracy: {baseline_acc}%\n")

    print("Legend:")
    print("  ✓ = Performs better than baseline (+5% or more)")
    print("  ⚠️  = Performs worse than baseline (-5% or more)")
    print("  (space) = Within ±5% of baseline\n")

    # Show key stats
    worst_bucket = df.loc[df["accuracy_pct"].idxmin()]
    best_bucket = df.loc[df["accuracy_pct"].idxmax()]

    print(f"\n🚨 WORST BUCKET:")
    print(f"   {worst_bucket['metric']} - {worst_bucket['bucket']}")
    print(
        f"   {worst_bucket['accuracy_pct']:.1f}% accuracy ({worst_bucket['vs_baseline']:+.1f}) - N={worst_bucket['total']}"
    )

    print(f"\n✅ BEST BUCKET:")
    print(f"   {best_bucket['metric']} - {best_bucket['bucket']}")
    print(
        f"   {best_bucket['accuracy_pct']:.1f}% accuracy ({best_bucket['vs_baseline']:+.1f}) - N={best_bucket['total']}"
    )

    # Summary by metric
    print(f"\n📊 METRIC SUMMARY:")
    metric_summary = (
        df.groupby("metric")
        .agg(
            {
                "accuracy_pct": ["mean", "min", "max"],
                "vs_baseline": "mean",
                "total": "sum",
            }
        )
        .round(2)
    )
    metric_summary.columns = [
        "Avg_Acc",
        "Min_Acc",
        "Max_Acc",
        "Avg_vs_Base",
        "Total_Games",
    ]
    metric_summary = metric_summary.sort_values("Avg_vs_Base", ascending=False)

    print("\nMetrics performing above baseline on average:")
    above = metric_summary[metric_summary["Avg_vs_Base"] > 0].head(5)
    for metric, row in above.iterrows():
        print(
            f"  ✓ {metric:25s}: {row['Avg_Acc']:5.2f}% avg (range: {row['Min_Acc']:.1f}-{row['Max_Acc']:.1f}%)"
        )

    print("\nMetrics performing below baseline on average:")
    below = metric_summary[metric_summary["Avg_vs_Base"] < 0].tail(5)
    for metric, row in below.iterrows():
        print(
            f"  ⚠️  {metric:25s}: {row['Avg_Acc']:5.2f}% avg (range: {row['Min_Acc']:.1f}-{row['Max_Acc']:.1f}%)"
        )
