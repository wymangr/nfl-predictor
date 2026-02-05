"""Analyze past predictions using historical bucket performance for confidence validation."""

import sys
import io
import pandas as pd
from src.helpers.database_helpers import run_query
from src.reports.past_predictions_analysis import (
    get_bucket_analysis,
    classify_prediction_bucket,
    calculate_bucket_confidence,
)


def analyze_past_predictions_with_buckets():
    """
    Analyze past predictions by comparing model confidence to bucket-based confidence.
    Shows calibration of both confidence systems against actual outcomes.

    Returns:
        DataFrame with past predictions and their bucket-based confidence scores
    """
    print("Loading historical bucket analysis...")
    bucket_df = get_bucket_analysis("past_predictions")
    baseline_acc = bucket_df["baseline_accuracy"].iloc[0]

    print(f"Baseline accuracy: {baseline_acc}%\n")

    # Get past predictions
    print("Loading past predictions...")
    past_preds = run_query(
        """
    SELECT 
        game_id,
        season,
        week,
        home_team,
        away_team,
        spread,
        predicted_winner,
        predicted_diff,
        confidence_score,
        confidence,
        div_game,
        power_ranking_diff,
        home_qb_changed,
        away_qb_changed,
        avg_weekly_point_diff_l3,
        rushing_epa_diff,
        passing_epa_diff,
        yards_per_attempt_diff,
        yards_per_carry_diff,
        win_pct_diff,
        adj_overall_rank_diff,
        cover_spread_by,
        correct
    FROM past_predictions
    WHERE correct != 'push'
    ORDER BY season DESC, week DESC, game_id
    """
    )

    past_df = pd.DataFrame(past_preds)

    if len(past_df) == 0:
        print("No past predictions found.")
        return None

    print(f"Found {len(past_df)} past predictions.\n")

    # Analyze each prediction
    results = []

    for idx, row in past_df.iterrows():
        game_info = {
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "predicted_winner": row["predicted_winner"],
            "model_confidence": row["confidence_score"],
            "correct": row["correct"] == "1",
        }

        # Use shared function to calculate bucket confidence
        weighted_acc, bucket_details = calculate_bucket_confidence(row, bucket_df)

        # Calculate overall expected accuracy
        if weighted_acc is not None:
            # Calculate simple average for comparison
            simple_avg_acc = sum(
                [bd["historical_accuracy"] for bd in bucket_details]
            ) / len(bucket_details)

            game_info["bucket_based_confidence"] = round(weighted_acc, 2)
            game_info["simple_avg_confidence"] = round(simple_avg_acc, 2)
            game_info["vs_baseline"] = round(weighted_acc - baseline_acc, 2)
            game_info["num_buckets_analyzed"] = len(bucket_details)
            game_info["bucket_details"] = bucket_details

            results.append(game_info)

    return pd.DataFrame(results), baseline_acc


def print_past_bucket_analysis():
    """Print detailed analysis of past predictions using bucket-based confidence."""
    results_df, baseline_acc = analyze_past_predictions_with_buckets()

    if results_df is None or len(results_df) == 0:
        return

    print("=" * 100)
    print("PAST PREDICTIONS ANALYSIS - BUCKET-BASED CONFIDENCE FOR ALL GAMES")
    print("=" * 100)
    print(f"\nBaseline Historical Accuracy: {baseline_acc}%")
    print(f"Total Past Games Analyzed: {len(results_df)}\n")

    # Summary statistics
    avg_bucket_conf = results_df["bucket_based_confidence"].mean()
    avg_model_conf = results_df["model_confidence"].mean()
    actual_accuracy = (results_df["correct"].sum() / len(results_df)) * 100

    print(f"Average Model Confidence Score: {avg_model_conf:.2f}%")
    print(f"Average Bucket-Based Confidence: {avg_bucket_conf:.2f}%")
    print(f"Actual Accuracy: {actual_accuracy:.2f}%\n")

    # Show ALL games with their bucket-based confidence
    print("=" * 120)
    print("ALL PAST PREDICTIONS WITH BUCKET-BASED CONFIDENCE")
    print("=" * 120)
    print(
        f"\n{'Season':<6} {'Week':<5} {'Matchup':<28} {'Winner':<6} {'Result':<7} {'Model':<7} {'Bucket':<7} {'Delta':<7} {'Buckets':<8}"
    )
    print("-" * 120)

    # Sort by season and week
    results_sorted = results_df.sort_values(
        ["season", "week"], ascending=[False, False]
    )

    for idx, row in results_sorted.iterrows():
        result_icon = "OK" if row["correct"] else "NO"
        result_text = "CORRECT" if row["correct"] else "WRONG"

        # Show difference between model confidence and bucket accuracy
        delta = row["bucket_based_confidence"] - row["model_confidence"]

        print(
            f"{row['season']:<6} {row['week']:<5} {row['matchup']:<28} {row['predicted_winner']:<6} "
            f"{result_icon} {result_text:<6} {row['model_confidence']:>5.1f}% {row['bucket_based_confidence']:>5.1f}% "
            f"{delta:>+6.1f}% {row['num_buckets_analyzed']:>3} metrics"
        )

    print("\n" + "=" * 120)
    print("BUCKET CONFIDENCE SUMMARY")
    print("=" * 120)

    # Compare model confidence to bucket accuracy as indicators
    # When correct: higher confidence/accuracy is better
    # When wrong: lower confidence/accuracy is better
    bucket_better = 0
    model_better = 0

    for idx, row in results_df.iterrows():
        # If correct and bucket was lower, or wrong and bucket was lower, bucket was better indicator
        if row["correct"]:
            # Correct - should have high confidence
            if abs(row["bucket_based_confidence"] - 100) < abs(
                row["model_confidence"] - 100
            ):
                bucket_better += 1
            else:
                model_better += 1
        else:
            # Wrong - should have low confidence
            if abs(row["bucket_based_confidence"] - 0) < abs(
                row["model_confidence"] - 0
            ):
                bucket_better += 1
            else:
                model_better += 1

    print(
        f"\nHistorical bucket accuracy was closer to outcome: {bucket_better} games ({bucket_better/len(results_df)*100:.1f}%)"
    )
    print(
        f"Model confidence was closer to outcome: {model_better} games ({model_better/len(results_df)*100:.1f}%)"
    )

    # Calibration analysis
    print("\n" + "=" * 100)
    print("CONFIDENCE CALIBRATION")
    print("=" * 100)
    print(
        f"\nModel Confidence vs Actual: {avg_model_conf - actual_accuracy:+.2f}% (calibration error)"
    )
    print(
        f"Bucket Confidence vs Actual: {avg_bucket_conf - actual_accuracy:+.2f}% (calibration error)"
    )

    # Calculate Brier score (lower is better)
    model_brier = (
        (results_df["model_confidence"] / 100 - results_df["correct"]) ** 2
    ).mean()
    bucket_brier = (
        (results_df["bucket_based_confidence"] / 100 - results_df["correct"]) ** 2
    ).mean()

    print(f"\nBrier Score (lower is better):")
    print(f"  Model Confidence: {model_brier:.4f}")
    print(f"  Bucket Confidence: {bucket_brier:.4f}")
    if bucket_brier < model_brier:
        print(
            f"  >> Bucket-based confidence is better calibrated by {model_brier - bucket_brier:.4f}"
        )
    else:
        print(
            f"  >> Model confidence is better calibrated by {bucket_brier - model_brier:.4f}"
        )

    # Games above/below baseline
    above_baseline = len(
        results_df[results_df["bucket_based_confidence"] > baseline_acc]
    )
    below_baseline = len(
        results_df[results_df["bucket_based_confidence"] < baseline_acc]
    )

    print(
        f"\nGames w/ Above-Baseline Buckets: {above_baseline} ({above_baseline/len(results_df)*100:.1f}%)"
    )
    print(
        f"Games w/ Below-Baseline Buckets: {below_baseline} ({below_baseline/len(results_df)*100:.1f}%)"
    )

    # Accuracy by confidence bracket
    print("\n" + "=" * 100)
    print("ACCURACY BY BUCKET CONFIDENCE BRACKET (1% intervals)")
    print("=" * 100)

    print(
        f"\n{'Bucket Confidence Range':<30} | {'Games':>6} | {'Correct':>7} | {'Accuracy':>8}"
    )
    print("-" * 100)

    # Generate 1% brackets from 50% to 100%
    brackets = []
    for i in range(50, 100, 1):
        min_conf = i
        max_conf = i + 1
        if max_conf > 100:
            max_conf = 100
        label = f"{min_conf}-{max_conf}%"
        brackets.append((min_conf, max_conf, label))

    # Add a final bracket for anything below 50%
    brackets.insert(0, (0, 50, "Below 50%"))

    for min_conf, max_conf, label in brackets:
        bracket_games = results_df[
            (results_df["bucket_based_confidence"] >= min_conf)
            & (results_df["bucket_based_confidence"] < max_conf)
        ]

        if len(bracket_games) > 0:
            correct_count = bracket_games["correct"].sum()
            accuracy = (correct_count / len(bracket_games)) * 100
            print(
                f"{label:<30} | {len(bracket_games):>6} | {correct_count:>7} | {accuracy:>7.1f}%"
            )


if __name__ == "__main__":
    # Windows console UTF-8 encoding for emojis
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print_past_bucket_analysis()
