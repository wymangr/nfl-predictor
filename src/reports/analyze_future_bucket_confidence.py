"""Analyze future predictions using historical bucket performance."""

import pandas as pd
from src.helpers.database_helpers import run_query
from src.reports.past_predictions_analysis import (
    get_bucket_analysis,
    calculate_bucket_confidence,
)


def analyze_future_predictions():
    """
    Analyze future predictions by comparing them to historical bucket performance.

    Returns:
        DataFrame with future predictions and their bucket-based confidence scores
    """
    print("Loading historical bucket analysis...")
    bucket_df = get_bucket_analysis("past_predictions")
    baseline_acc = bucket_df["baseline_accuracy"].iloc[0]

    print(f"Baseline accuracy: {baseline_acc}%\n")

    # Get future predictions
    print("Loading future predictions...")
    future_preds = run_query(
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
        cover_spread_by
    FROM future_predictions
    ORDER BY week, game_id
    """
    )

    future_df = pd.DataFrame(future_preds)

    if len(future_df) == 0:
        print("No future predictions found.")
        return None

    print(f"Found {len(future_df)} future predictions.\n")

    # Analyze each prediction
    results = []

    for idx, row in future_df.iterrows():
        game_info = {
            "game_id": row["game_id"],
            "week": row["week"],
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "predicted_winner": row["predicted_winner"],
            "model_confidence": row["confidence_score"],
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


def print_future_analysis():
    """Print detailed analysis of future predictions."""
    results_df, baseline_acc = analyze_future_predictions()

    if results_df is None or len(results_df) == 0:
        return

    print("=" * 100)
    print("FUTURE PREDICTIONS ANALYSIS - BUCKET-BASED CONFIDENCE FOR ALL GAMES")
    print("=" * 100)
    print(f"\nBaseline Historical Accuracy: {baseline_acc}%")
    print(f"Total Future Games: {len(results_df)}\n")

    # Summary statistics
    avg_bucket_conf = results_df["bucket_based_confidence"].mean()
    avg_model_conf = results_df["model_confidence"].mean()

    print(f"Average Model Confidence Score: {avg_model_conf:.2f}%")
    print(f"Average Bucket-Based Confidence: {avg_bucket_conf:.2f}%")
    print(f"Average vs Baseline: {avg_bucket_conf - baseline_acc:+.2f}%\n")

    # Show ALL future games with their bucket-based confidence
    print("=" * 120)
    print("ALL FUTURE PREDICTIONS WITH BUCKET-BASED CONFIDENCE")
    print("=" * 120)
    print(
        f"\n{'Week':<5} {'Matchup':<28} {'Winner':<6} {'Model':<7} {'Bucket':<7} {'Delta':<7} {'vs Base':<8} {'Buckets':<8}"
    )
    print("-" * 120)

    # Sort by week
    results_sorted = results_df.sort_values("week", ascending=True)

    for idx, row in results_sorted.iterrows():
        delta = row["bucket_based_confidence"] - row["model_confidence"]

        print(
            f"{row['week']:<5} {row['matchup']:<28} {row['predicted_winner']:<6} "
            f"{row['model_confidence']:>5.1f}% {row['bucket_based_confidence']:>5.1f}% "
            f"{delta:>+6.1f}% {row['vs_baseline']:>+6.1f}% {row['num_buckets_analyzed']:>3} metrics"
        )

    print("\n" + "=" * 120)
    print("BUCKET CONFIDENCE SUMMARY")
    print("=" * 120)

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

    # Show difference between model and bucket confidence
    higher_model = len(
        results_df[
            results_df["model_confidence"] > results_df["bucket_based_confidence"]
        ]
    )
    print(
        f"\nGames where model confidence > historical bucket accuracy: {higher_model} ({higher_model/len(results_df)*100:.1f}%)"
    )

    avg_difference = (
        results_df["model_confidence"] - results_df["bucket_based_confidence"]
    ).mean()
    print(f"Average difference (Model - Bucket): {avg_difference:+.2f}%")

    # Sort by bucket-based confidence
    results_df_sorted = results_df.sort_values(
        "bucket_based_confidence", ascending=False
    )

    print("\n" + "=" * 100)
    print("DETAILED GAME ANALYSIS - HIGHEST CONFIDENCE GAMES")
    print("=" * 100)

    num_to_show = min(5, len(results_df_sorted))
    for idx, row in results_df_sorted.head(num_to_show).iterrows():
        print(f"\nWeek {row['week']:2d} | {row['matchup']:25s}")
        print(f"  Predicted Winner: {row['predicted_winner']}")
        print(f"  Model Confidence: {row['model_confidence']:.1f}%")
        print(
            f"  Bucket-Based Confidence: {row['bucket_based_confidence']:.1f}% ({row['vs_baseline']:+.1f} vs baseline)"
        )
        print(f"  Analyzed {row['num_buckets_analyzed']} metric buckets")

        # Show top 3 strongest/weakest indicators
        details = sorted(
            row["bucket_details"], key=lambda x: x["vs_baseline"], reverse=True
        )
        if len(details) >= 3:
            print(f"  Strongest indicators:")
            for d in details[:3]:
                if d["vs_baseline"] > 0:
                    print(
                        f"    + {d['metric']:20s}: {d['bucket']:30s} ({d['historical_accuracy']:.1f}%, +{d['vs_baseline']:.1f})"
                    )

        weak_details = sorted(row["bucket_details"], key=lambda x: x["vs_baseline"])
        if len(weak_details) >= 1 and weak_details[0]["vs_baseline"] < -5:
            print(f"  Warning indicators:")
            for d in weak_details[:3]:
                if d["vs_baseline"] < -5:
                    print(
                        f"    ! {d['metric']:20s}: {d['bucket']:30s} ({d['historical_accuracy']:.1f}%, {d['vs_baseline']:.1f})"
                    )

    print("\n" + "=" * 100)
    print("DETAILED GAME ANALYSIS - LOWEST CONFIDENCE GAMES (RISKY)")
    print("=" * 100)

    num_to_show = min(5, len(results_df_sorted))
    for idx, row in results_df_sorted.tail(num_to_show).iterrows():
        print(f"\nWeek {row['week']:2d} | {row['matchup']:25s}")
        print(f"  Predicted Winner: {row['predicted_winner']}")
        print(f"  Model Confidence: {row['model_confidence']:.1f}%")
        print(
            f"  Bucket-Based Confidence: {row['bucket_based_confidence']:.1f}% ({row['vs_baseline']:+.1f} vs baseline)"
        )
        print(f"  Analyzed {row['num_buckets_analyzed']} metric buckets")

        # Show weakest indicators
        details = sorted(row["bucket_details"], key=lambda x: x["vs_baseline"])
        if len(details) >= 3:
            print(f"  Warning indicators:")
            for d in details[:5]:
                marker = "! " if d["vs_baseline"] < -5 else "  "
                print(
                    f"    {marker}{d['metric']:20s}: {d['bucket']:30s} ({d['historical_accuracy']:.1f}%, {d['vs_baseline']:+.1f})"
                )

    print("\n" + "=" * 100)
    print("OVERALL EXPECTED ACCURACY")
    print("=" * 100)
    print(f"\nWeighted by bucket sample sizes:")
    print(f"  Expected Accuracy: {avg_bucket_conf:.2f}%")
    print(f"  vs Baseline: {avg_bucket_conf - baseline_acc:+.2f}%")
    print(f"  vs Model Confidence: {avg_bucket_conf - avg_model_conf:+.2f}%")

    # Confidence brackets with 1% intervals
    print("\n" + "=" * 100)
    print("CONFIDENCE DISTRIBUTION (1% intervals)")
    print("=" * 100)

    print(f"\n{'Bucket Confidence Range':<30} | {'Games':>6}")
    print("-" * 100)

    # Generate 1% brackets from 50% to 100%
    brackets = []
    for i in range(50, 100, 1):
        min_conf = i
        max_conf = i + 1
        label = f"{min_conf}-{max_conf}%"
        brackets.append((min_conf, max_conf, label))

    # Add a bracket for anything below 50%
    brackets.insert(0, (0, 50, "Below 50%"))

    for min_conf, max_conf, label in brackets:
        count = len(
            results_df[
                (results_df["bucket_based_confidence"] >= min_conf)
                & (results_df["bucket_based_confidence"] < max_conf)
            ]
        )
        if count > 0:
            print(f"{label:<30} | {count:>6}")
