import pandas as pd
from datetime import datetime
from sqlalchemy import text

from src.helpers.database_helpers import get_db_engine


def load_data():
    """Load predictions from database and join with games data."""
    # Join past_predictions with games table
    engine = get_db_engine()
    query = """
    SELECT 
        pp.*,
        g.div_game,
        g.gametime,
        g.weekday,
        g.away_rest,
        g.home_rest,
        g.roof,
        g.surface,
        g.temp,
        g.wind,
        g.away_qb_name,
        g.home_qb_name,
        g.total_line,
        g.away_score + g.home_score as total_score
    FROM past_predictions pp
    LEFT JOIN games g ON pp.game_id = g.game_id
    WHERE pp.correct != 'push'
    """

    df = pd.read_sql_query(query, engine)

    # Convert correct column to numeric (it comes as strings "1"/"0")
    df["correct"] = pd.to_numeric(df["correct"])

    # Handle duplicate columns by removing them
    df = df.loc[:, ~df.columns.duplicated()]

    # Add derived columns
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["is_primetime"] = df["gametime"].apply(
        lambda x: x >= "20:00" if pd.notna(x) else False
    )
    df["is_thursday"] = df["weekday"] == "Thursday"
    df["high_wind"] = df["wind"] > 15
    df["cold_weather"] = df["temp"] < 32
    df["outdoor"] = df["roof"] == "outdoors"

    return df


def calculate_overall_stats(df):
    """Calculate overall prediction statistics."""
    total_predictions = len(df)
    correct_predictions = df["correct"].sum()
    accuracy = (
        (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    )

    return {
        "total": total_predictions,
        "correct": correct_predictions,
        "incorrect": total_predictions - correct_predictions,
        "accuracy": accuracy,
    }


def analyze_by_team(df):
    """Analyze performance for each team (both home and away)."""
    all_home = df[["home_team", "correct"]].rename(columns={"home_team": "team"})
    all_away = df[["away_team", "correct"]].rename(columns={"away_team": "team"})
    combined = pd.concat([all_home, all_away])

    team_stats = (
        combined.groupby("team")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    team_stats.columns = ["team", "correct", "total", "accuracy"]
    team_stats["accuracy"] = team_stats["accuracy"] * 100
    team_stats = team_stats.sort_values("accuracy", ascending=False)

    return team_stats


def analyze_spread_favorite_location(df):
    """Analyze performance when spread favors home vs away team."""
    df["fav_home"] = df["spread_favorite"] == df["home_team"]

    stats = (
        df.groupby("fav_home").agg({"correct": ["sum", "count", "mean"]}).reset_index()
    )
    stats.columns = ["fav_home", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["favorite"] = stats["fav_home"].map({True: "Home Team", False: "Away Team"})

    return stats[["favorite", "correct", "total", "accuracy"]]


def analyze_home_away_picks(df):
    """Analyze performance when model picks home vs away team."""
    df["picked_home"] = df["predicted_winner"] == df["home_team"]

    stats = (
        df.groupby("picked_home")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["picked_home", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["pick_location"] = stats["picked_home"].map(
        {True: "Home Team", False: "Away Team"}
    )

    return stats[["pick_location", "correct", "total", "accuracy"]]


def analyze_by_spread_magnitude(df):
    """Analyze performance by spread size."""
    df["spread_bucket"] = pd.cut(
        df["spread"],
        bins=[0, 3, 7, 14, 100],
        labels=["0-3 (Close)", "3-7 (Medium)", "7-14 (Large)", "14+ (Blowout)"],
    )

    stats = (
        df.groupby("spread_bucket", observed=True)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["spread_range", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_confidence_performance(df):
    """Analyze how confidence correlates with accuracy - grouped by confidence rank."""
    max_conf = df["confidence"].max()

    # Group by confidence rank ranges (in points)
    # Note: confidence rank 1 = lowest confidence, max rank = highest confidence
    df["conf_bucket"] = pd.cut(
        df["confidence"],
        bins=[0, max_conf * 0.25, max_conf * 0.5, max_conf * 0.75, max_conf],
        labels=[
            f"Rank 1-{int(max_conf * 0.25)} (Low)",
            f"Rank {int(max_conf * 0.25) + 1}-{int(max_conf * 0.5)} (Med-Low)",
            f"Rank {int(max_conf * 0.5) + 1}-{int(max_conf * 0.75)} (Med-High)",
            f"Rank {int(max_conf * 0.75) + 1}-{int(max_conf)} (High)",
        ],
    )

    stats = (
        df.groupby("conf_bucket", observed=True)
        .agg({"correct": ["sum", "count", "mean"], "cover_spread_by": "mean"})
        .reset_index()
    )
    stats.columns = ["confidence_level", "correct", "total", "accuracy", "avg_margin"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_favorite_vs_underdog(df):
    """Analyze accuracy when picking favorite vs underdog."""
    df["picked_favorite"] = df["predicted_winner"] == df["spread_favorite"]

    stats = (
        df.groupby("picked_favorite")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["picked_favorite", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["pick_type"] = stats["picked_favorite"].map(
        {True: "Favorite", False: "Underdog"}
    )

    return stats[["pick_type", "correct", "total", "accuracy"]]


def analyze_by_week(df):
    """Analyze performance by week."""
    stats = df.groupby("week").agg({"correct": ["sum", "count", "mean"]}).reset_index()
    stats.columns = ["week", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_confidence_points_by_week(df):
    """Analyze total confidence points earned by week (only for correct predictions)."""
    # Filter to only correct predictions
    correct_df = df[df["correct"] == 1].copy()

    # Group by week and sum confidence points
    stats = (
        correct_df.groupby("week")
        .agg({"confidence": "sum", "correct": "count"})
        .reset_index()
    )

    stats.columns = ["week", "confidence_points", "correct_picks"]

    return stats


def analyze_division_games(df):
    """Analyze performance on division games vs non-division games."""
    # Check if div_game column exists and has valid data
    if "div_game" not in df.columns or df["div_game"].isna().all():
        return pd.DataFrame(columns=["game_type", "correct", "total", "accuracy"])

    stats = (
        df.groupby("div_game").agg({"correct": ["sum", "count", "mean"]}).reset_index()
    )
    stats.columns = ["div_game", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["game_type"] = stats["div_game"].map({1: "Division Game", 0: "Non-Division"})

    return stats[["game_type", "correct", "total", "accuracy"]]


def analyze_primetime_games(df):
    """Analyze performance on prime time games."""
    stats = (
        df.groupby("is_primetime")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["is_primetime", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["game_type"] = stats["is_primetime"].map(
        {True: "Prime Time", False: "Regular"}
    )

    return stats[["game_type", "correct", "total", "accuracy"]]


def analyze_thursday_games(df):
    """Analyze performance on Thursday night games."""
    thursday = df[df["is_thursday"] == True]
    if len(thursday) == 0:
        return None

    total = len(thursday)
    correct = thursday["correct"].sum()
    accuracy = (correct / total * 100) if total > 0 else 0

    return {"correct": correct, "total": total, "accuracy": accuracy}


def analyze_weather_impact(df):
    """Analyze performance in different weather conditions."""
    conditions = []

    # High wind
    high_wind = df[df["high_wind"] == True]
    if len(high_wind) > 0:
        conditions.append(
            {
                "condition": "High Wind (>15 mph)",
                "correct": high_wind["correct"].sum(),
                "total": len(high_wind),
                "accuracy": (high_wind["correct"].sum() / len(high_wind) * 100),
            }
        )

    # Cold weather
    cold = df[df["cold_weather"] == True]
    if len(cold) > 0:
        conditions.append(
            {
                "condition": "Cold (<32°F)",
                "correct": cold["correct"].sum(),
                "total": len(cold),
                "accuracy": (cold["correct"].sum() / len(cold) * 100),
            }
        )

    # Outdoor vs Indoor
    outdoor_stats = (
        df.groupby("outdoor").agg({"correct": ["sum", "count", "mean"]}).reset_index()
    )
    outdoor_stats.columns = ["outdoor", "correct", "total", "accuracy"]
    outdoor_stats["accuracy"] = outdoor_stats["accuracy"] * 100
    outdoor_stats["condition"] = outdoor_stats["outdoor"].map(
        {True: "Outdoor", False: "Indoor/Dome"}
    )

    for _, row in outdoor_stats.iterrows():
        conditions.append(
            {
                "condition": row["condition"],
                "correct": row["correct"],
                "total": row["total"],
                "accuracy": row["accuracy"],
            }
        )

    return pd.DataFrame(conditions) if conditions else None


def analyze_rest_differential(df):
    """Analyze performance based on rest advantage."""
    df["rest_bucket"] = pd.cut(
        df["rest_diff"],
        bins=[-100, -3, -1, 1, 3, 100],
        labels=["Away +3 days", "Away +1-2", "Even", "Home +1-2", "Home +3 days"],
    )

    stats = (
        df.groupby("rest_bucket", observed=True)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["rest_advantage", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_close_games(df):
    """Analyze performance on close games (margin < 3)."""
    close = df[df["cover_spread_by"].abs() < 3]
    other = df[df["cover_spread_by"].abs() >= 3]

    stats = []
    for name, subset in [("Close (<3 pts)", close), ("Not Close (≥3 pts)", other)]:
        if len(subset) > 0:
            stats.append(
                {
                    "game_type": name,
                    "correct": subset["correct"].sum(),
                    "total": len(subset),
                    "accuracy": (subset["correct"].sum() / len(subset) * 100),
                }
            )

    return pd.DataFrame(stats)


def analyze_recent_form_impact(df):
    """Analyze when recent form (L3) diverges from overall stats."""
    # When recent form strongly favors one team but model picks other
    df["recent_form_signal"] = df["avg_weekly_point_diff_l3"].abs() > 10

    stats = (
        df.groupby("recent_form_signal")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["strong_recent_form", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["category"] = stats["strong_recent_form"].map(
        {True: "Strong Recent Form (L3 diff >10)", False: "Normal Recent Form"}
    )

    return stats[["category", "correct", "total", "accuracy"]]


def analyze_epa_mismatches(df):
    """Analyze games with EPA mismatches (pass vs rush)."""
    # When passing and rushing EPA tell different stories
    df["pass_rush_mismatch"] = (
        (df["passing_epa_diff"] > 50) & (df["rushing_epa_diff"] < -50)
    ) | ((df["passing_epa_diff"] < -50) & (df["rushing_epa_diff"] > 50))

    stats = (
        df.groupby("pass_rush_mismatch")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["has_mismatch", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats["category"] = stats["has_mismatch"].map(
        {True: "Pass/Rush EPA Mismatch", False: "EPA Aligned"}
    )

    return stats[["category", "correct", "total", "accuracy"]]


def analyze_by_margin_magnitude(df):
    """Analyze performance by predicted margin size."""
    df_copy = df.copy()
    df_copy["margin_bucket"] = pd.cut(
        df_copy["cover_spread_by"].abs(),
        bins=[0, 3, 7, 14, 100],
        labels=["0-3 (Close)", "3-7 (Medium)", "7-14 (Large)", "14+ (Blowout)"],
    )

    stats = (
        df_copy.groupby("margin_bucket", observed=True)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["margin_range", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_power_ranking_diff(df):
    """Analyze performance by power ranking differential magnitude."""
    df_copy = df[df["power_ranking_diff"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(columns=["power_rank_diff", "correct", "total", "accuracy"])

    # Higher absolute value = bigger mismatch
    df_copy["power_rank_bucket"] = pd.cut(
        df_copy["power_ranking_diff"].abs(),
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=[
            "0-0.2 (Low)",
            "0.2-0.4 (Medium)",
            "0.4-0.6 (High)",
            "0.6+ (Very High)",
        ],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("power_rank_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["power_rank_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    # Filter out empty buckets
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_adj_overall_rank_diff(df):
    """Analyze performance by adjusted overall rank differential magnitude."""
    df_copy = df[df["adj_overall_rank_diff"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(columns=["rank_diff", "correct", "total", "accuracy"])

    # Higher absolute value = bigger mismatch
    df_copy["rank_bucket"] = pd.cut(
        df_copy["adj_overall_rank_diff"].abs(),
        bins=[0, 8, 16, 24, 50],
        labels=["0-8 (Low)", "8-16 (Medium)", "16-24 (High)", "24+ (Very High)"],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("rank_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["rank_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    # Filter out empty buckets
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_predicted_team(df):
    """Analyze performance for each team when the model picks them to win."""
    # Filter to only rows where this team was predicted to win
    team_as_predicted = df[df["predicted_winner"].notna()].copy()

    stats = (
        team_as_predicted.groupby("predicted_winner")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["team", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats.sort_values("accuracy", ascending=False)

    return stats


def analyze_by_power_ranking_diff_l3(df):
    """Analyze performance by recent power ranking differential (L3 games)."""
    # Check if column exists
    if "power_ranking_diff_l3" not in df.columns:
        return pd.DataFrame(
            columns=["power_rank_diff_l3", "correct", "total", "accuracy"]
        )

    df_copy = df[df["power_ranking_diff_l3"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(
            columns=["power_rank_diff_l3", "correct", "total", "accuracy"]
        )

    # Similar bins to power_ranking_diff but for L3
    df_copy["power_rank_l3_bucket"] = pd.cut(
        df_copy["power_ranking_diff_l3"].abs(),
        bins=[0, 0.2, 0.4, 1.0],
        labels=[
            "0-0.2 (Low)",
            "0.2-0.4 (Medium)",
            "0.4+ (High)",
        ],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("power_rank_l3_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["power_rank_diff_l3", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_avg_margin_of_victory_diff(df):
    """Analyze performance by average margin of victory differential."""
    # Check if column exists
    if "avg_margin_of_victory_diff" not in df.columns:
        return pd.DataFrame(columns=["mov_diff", "correct", "total", "accuracy"])

    df_copy = df[df["avg_margin_of_victory_diff"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(columns=["mov_diff", "correct", "total", "accuracy"])

    # Bin by margin of victory differential
    df_copy["mov_bucket"] = pd.cut(
        df_copy["avg_margin_of_victory_diff"].abs(),
        bins=[0, 3, 7, 14, 100],
        labels=["0-3 (Close)", "3-7 (Medium)", "7-14 (Large)", "14+ (Dominant)"],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("mov_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["mov_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_avg_margin_of_victory_diff_l3(df):
    """Analyze performance by recent average margin of victory differential (L3)."""
    # Check if column exists
    if "avg_margin_of_victory_diff_l3" not in df.columns:
        return pd.DataFrame(columns=["mov_diff_l3", "correct", "total", "accuracy"])

    df_copy = df[df["avg_margin_of_victory_diff_l3"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(columns=["mov_diff_l3", "correct", "total", "accuracy"])

    df_copy["mov_l3_bucket"] = pd.cut(
        df_copy["avg_margin_of_victory_diff_l3"].abs(),
        bins=[0, 3, 7, 14, 100],
        labels=["0-3 (Close)", "3-7 (Medium)", "7-14 (Large)", "14+ (Dominant)"],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("mov_l3_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["mov_diff_l3", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_avg_weekly_point_diff(df):
    """Analyze performance by average weekly point differential - L3."""
    # Use L3 version (3-game rolling average) for better signal
    if "avg_weekly_point_diff_l3" not in df.columns:
        return pd.DataFrame(columns=["weekly_pt_diff", "correct", "total", "accuracy"])

    df_copy = df[df["avg_weekly_point_diff_l3"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(columns=["weekly_pt_diff", "correct", "total", "accuracy"])

    # Create bins - OPTIMIZED for 24.3% variance (was 16.7%)
    df_copy["weekly_pt_bucket"] = pd.cut(
        df_copy["avg_weekly_point_diff_l3"].abs(),
        bins=[0, 5.667, 9.333, 100],
        labels=["0-5.7 (Even)", "5.7-9.3 (Moderate)", "9.3+ (Strong)"],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("weekly_pt_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["weekly_pt_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_spread_performance_diff(df):
    """Analyze performance by spread performance differential (ATS record)."""
    # Check if column exists
    if "spread_performance_diff" not in df.columns:
        return pd.DataFrame(
            columns=["spread_perf_diff", "correct", "total", "accuracy"]
        )

    df_copy = df[df["spread_performance_diff"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(
            columns=["spread_perf_diff", "correct", "total", "accuracy"]
        )

    # Create bins - OPTIMIZED for 23.8% variance (was 0%!)
    df_copy["spread_perf_bucket"] = pd.cut(
        df_copy["spread_performance_diff"],
        bins=[-float("inf"), -7.0, -3.5, -0.559, 1.722, 4.969, float("inf")],
        labels=[
            "< -7.0 (Very Poor ATS)",
            "-7.0 to -3.5 (Poor ATS)",
            "-3.5 to -0.6 (Below Avg)",
            "-0.6 to 1.7 (Avg)",
            "1.7 to 5.0 (Good ATS)",
            "> 5.0 (Elite ATS)",
        ],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("spread_perf_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["spread_perf_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats[stats["total"] > 0]

    return stats


def analyze_by_sacks_suffered_diff(df):
    """Analyze performance by sacks suffered differential (defensive pressure)."""
    # Check if column exists
    if "sacks_suffered_avg_diff" not in df.columns:
        return pd.DataFrame(columns=["sacks_diff", "correct", "total", "accuracy"])

    df_copy = df[df["sacks_suffered_avg_diff"].notna()].copy()

    if len(df_copy) == 0:
        return pd.DataFrame(columns=["sacks_diff", "correct", "total", "accuracy"])

    df_copy["sacks_bucket"] = pd.cut(
        df_copy["sacks_suffered_avg_diff"].abs(),
        bins=[0, 0.5, 1.0, 1.5, 10.0],
        labels=[
            "0-0.5 (Even)",
            "0.5-1.0 (Moderate)",
            "1.0-1.5 (Large)",
            "1.5+ (Extreme)",
        ],
        include_lowest=True,
    )

    stats = (
        df_copy.groupby("sacks_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["sacks_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100
    stats = stats[stats["total"] > 0]

    return stats


def analyze_qb_changes(df):
    """Analyze performance when quarterbacks change."""
    # Filter only games where we have QB change data
    df_with_qb = df[
        df["home_qb_changed"].notna() & df["away_qb_changed"].notna()
    ].copy()

    if len(df_with_qb) == 0:
        return None

    # Create categories
    df_with_qb["qb_change_category"] = "No QB Change"
    df_with_qb.loc[
        (df_with_qb["home_qb_changed"] == "Y") & (df_with_qb["away_qb_changed"] == "N"),
        "qb_change_category",
    ] = "Home QB Changed"
    df_with_qb.loc[
        (df_with_qb["home_qb_changed"] == "N") & (df_with_qb["away_qb_changed"] == "Y"),
        "qb_change_category",
    ] = "Away QB Changed"
    df_with_qb.loc[
        (df_with_qb["home_qb_changed"] == "Y") & (df_with_qb["away_qb_changed"] == "Y"),
        "qb_change_category",
    ] = "Both QBs Changed"

    stats = (
        df_with_qb.groupby("qb_change_category")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["category", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    # Order the categories
    category_order = [
        "No QB Change",
        "Home QB Changed",
        "Away QB Changed",
        "Both QBs Changed",
    ]
    stats["category"] = pd.Categorical(
        stats["category"], categories=category_order, ordered=True
    )
    stats = stats.sort_values("category")

    return stats


def analyze_qb_change_any(df):
    """Analyze performance: QB change vs no QB change (simplified)."""
    # Filter only games where we have QB change data
    df_with_qb = df[
        df["home_qb_changed"].notna() & df["away_qb_changed"].notna()
    ].copy()

    if len(df_with_qb) == 0:
        return None

    # Create binary category: any QB change vs no change
    df_with_qb["qb_change_any"] = "No QB Change"
    df_with_qb.loc[
        (df_with_qb["home_qb_changed"] == "Y") | (df_with_qb["away_qb_changed"] == "Y"),
        "qb_change_any",
    ] = "QB Change (Any)"

    stats = (
        df_with_qb.groupby("qb_change_any")
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["category", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    # Order the categories
    category_order = ["No QB Change", "QB Change (Any)"]
    stats["category"] = pd.Categorical(
        stats["category"], categories=category_order, ordered=True
    )
    stats = stats.sort_values("category")

    return stats


def analyze_by_completion_pct_diff(df):
    """Analyze performance by completion percentage differential."""
    # Check if completion_pct_diff exists
    if "completion_pct_diff" not in df.columns:
        return pd.DataFrame()

    # Filter valid data
    valid_df = df[pd.notna(df["completion_pct_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins for completion percentage differential
    bins = [-float("inf"), -0.10, -0.05, 0.05, 0.10, float("inf")]
    labels = ["< -10%", "-10% to -5%", "-5% to 5%", "5% to 10%", "> 10%"]

    valid_df["completion_pct_diff_bucket"] = pd.cut(
        valid_df["completion_pct_diff"].abs(), bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("completion_pct_diff_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["completion_pct_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_yards_per_attempt_diff(df):
    """Analyze performance by yards per attempt differential."""
    # Check if yards_per_attempt_diff exists
    if "yards_per_attempt_diff" not in df.columns:
        return pd.DataFrame()

    # Filter valid data
    valid_df = df[pd.notna(df["yards_per_attempt_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins for yards per attempt differential - OPTIMIZED for 19% variance
    bins = [-float("inf"), -1.542, -0.744, -0.013, 0.501, 1.341, float("inf")]
    labels = [
        "< -1.54 (Very Weak)",
        "-1.54 to -0.74 (Weak)",
        "-0.74 to 0 (Below Avg)",
        "0 to 0.50 (Above Avg)",
        "0.50 to 1.34 (Strong)",
        "> 1.34 (Elite)",
    ]

    valid_df["yards_per_attempt_diff_bucket"] = pd.cut(
        valid_df["yards_per_attempt_diff"].abs(), bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("yards_per_attempt_diff_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["yards_per_attempt_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_yards_per_carry_diff(df):
    """Analyze performance by yards per carry differential."""
    # Check if yards_per_carry_diff exists
    if "yards_per_carry_diff" not in df.columns:
        return pd.DataFrame()

    # Filter valid data
    valid_df = df[pd.notna(df["yards_per_carry_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins for yards per carry differential - OPTIMIZED for 30% variance
    bins = [-float("inf"), -0.747, -0.163, 0.351, 0.869, float("inf")]
    labels = [
        "< -0.75 (Very Weak)",
        "-0.75 to -0.16 (Weak)",
        "-0.16 to 0.35 (Even)",
        "0.35 to 0.87 (Strong)",
        "> 0.87 (Dominant)",
    ]

    valid_df["yards_per_carry_diff_bucket"] = pd.cut(
        valid_df["yards_per_carry_diff"].abs(), bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("yards_per_carry_diff_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["yards_per_carry_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_spread_diff(df):
    """Analyze performance by spread differential (difference between sportsbooks)."""
    if "spread_diff" not in df.columns:
        return pd.DataFrame()

    valid_df = df[pd.notna(df["spread_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Take absolute value for bucketing
    valid_df["spread_diff_abs"] = valid_df["spread_diff"].abs()

    # Create bins: 0, 0.5-2, 2.5-4
    bins = [-float("inf"), 0.25, 2.25, 4, float("inf")]
    labels = ["0", "0.5-2", "2.5-4", ">4"]

    valid_df["spread_diff_bucket"] = pd.cut(
        valid_df["spread_diff_abs"], bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("spread_diff_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["spread_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_rushing_yards_diff(df):
    """Analyze performance by rushing yards differential (30% variance - TOP METRIC!)."""
    if "rushing_yards_diff" not in df.columns:
        return pd.DataFrame()

    valid_df = df[pd.notna(df["rushing_yards_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins based on outlier analysis results
    bins = [-float("inf"), -100, -25, 25, 100, float("inf")]
    labels = [
        "< -100 (Very Weak)",
        "-100 to -25 (Weak)",
        "-25 to 25 (Even)",
        "25 to 100 (Strong)",
        "> 100 (Dominant)",
    ]

    valid_df["rushing_yards_bucket"] = pd.cut(
        valid_df["rushing_yards_diff"], bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("rushing_yards_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["rushing_yards_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_def_rushing_yards_diff(df):
    """Analyze performance by defensive rushing yards differential (13.9% variance)."""
    if "def_rushing_yards_diff" not in df.columns:
        return pd.DataFrame()

    valid_df = df[pd.notna(df["def_rushing_yards_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins - defensive stats (negative is better)
    bins = [-float("inf"), -150, -50, 50, 150, float("inf")]
    labels = [
        "< -150 (Elite D)",
        "-150 to -50 (Good D)",
        "-50 to 50 (Average)",
        "50 to 150 (Poor D)",
        "> 150 (Very Poor D)",
    ]

    valid_df["def_rushing_yards_bucket"] = pd.cut(
        valid_df["def_rushing_yards_diff"], bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("def_rushing_yards_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["def_rushing_yards_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_passing_yards_diff(df):
    """Analyze performance by passing yards differential (16.9% variance)."""
    if "passing_yards_diff" not in df.columns:
        return pd.DataFrame()

    valid_df = df[pd.notna(df["passing_yards_diff"])].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins
    bins = [-float("inf"), -200, -75, 75, 200, float("inf")]
    labels = [
        "< -200 (Very Weak)",
        "-200 to -75 (Weak)",
        "-75 to 75 (Even)",
        "75 to 200 (Strong)",
        "> 200 (Dominant)",
    ]

    valid_df["passing_yards_bucket"] = pd.cut(
        valid_df["passing_yards_diff"], bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("passing_yards_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["passing_yards_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def analyze_by_turnover_diff(df):
    """Analyze performance by total turnover differential (INT + Fumbles)."""
    # Check if turnover_diff column already exists (from predict.py)
    if "turnover_diff" in df.columns:
        valid_df = df.copy()
        # Filter out NaN values
        valid_df = valid_df[pd.notna(valid_df["turnover_diff"])]
    else:
        # Try to calculate from individual columns if available
        required_cols = [
            "passing_interceptions_diff",
            "sack_fumbles_lost_diff",
            "rushing_fumbles_lost_diff",
            "def_passing_interceptions_diff",
            "def_sack_fumbles_lost_diff",
            "def_rushing_fumbles_lost_diff",
        ]

        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        # Calculate total turnover differential
        valid_df = df.copy()

        # Team's turnovers committed
        valid_df["turnovers_committed"] = (
            valid_df["passing_interceptions_diff"]
            + valid_df["sack_fumbles_lost_diff"]
            + valid_df["rushing_fumbles_lost_diff"]
        )

        # Team's turnovers forced
        valid_df["turnovers_forced"] = (
            valid_df["def_passing_interceptions_diff"]
            + valid_df["def_sack_fumbles_lost_diff"]
            + valid_df["def_rushing_fumbles_lost_diff"]
        )

        # Net turnover differential (positive means more takeaways than giveaways)
        valid_df["turnover_diff"] = (
            valid_df["turnovers_forced"] - valid_df["turnovers_committed"]
        )

        # Filter out NaN values
        valid_df = valid_df[pd.notna(valid_df["turnover_diff"])]

    if len(valid_df) == 0:
        return pd.DataFrame()

    # Create bins for turnover differential
    bins = [-float("inf"), -2, -1, 0, 1, 2, float("inf")]
    labels = ["<= -2", "-1", "0 (Even)", "+1", "+2", ">= +3"]

    valid_df["turnover_diff_bucket"] = pd.cut(
        valid_df["turnover_diff"], bins=bins, labels=labels
    )

    stats = (
        valid_df.groupby("turnover_diff_bucket", observed=False)
        .agg({"correct": ["sum", "count", "mean"]})
        .reset_index()
    )
    stats.columns = ["turnover_diff", "correct", "total", "accuracy"]
    stats["accuracy"] = stats["accuracy"] * 100

    return stats


def find_outliers(df):
    """Find notable outliers."""
    wrong_high_conf = (
        df[(df["correct"] == 0) & (df["confidence"] >= df["confidence"].quantile(0.75))]
        .sort_values("confidence", ascending=False)
        .head(10)
    )
    right_low_conf = (
        df[(df["correct"] == 1) & (df["confidence"] <= df["confidence"].quantile(0.25))]
        .sort_values("confidence")
        .head(10)
    )

    # Large spread underdogs that won
    upset_wins = (
        df[
            (df["correct"] == 1)
            & (df["predicted_winner"] != df["spread_favorite"])
            & (df["spread"] > 7)
        ]
        .sort_values("spread", ascending=False)
        .head(10)
    )

    return wrong_high_conf, right_low_conf, upset_wins


def get_accuracy_class(accuracy):
    """Return CSS class based on accuracy level."""
    if accuracy >= 60:
        return "accuracy-good"
    elif accuracy < 50:
        return "accuracy-bad"
    else:
        return "accuracy-medium"


def generate_past_prediction_report(df, output_file="nfl_past_prediction_report.html"):
    """Generate comprehensive HTML report."""

    # Calculate all statistics
    overall = calculate_overall_stats(df)
    team_stats = analyze_by_team(df)
    spread_location = analyze_spread_favorite_location(df)
    home_away_picks = analyze_home_away_picks(df)
    spread_magnitude = analyze_by_spread_magnitude(df)
    confidence_perf = analyze_confidence_performance(df)
    fav_vs_dog = analyze_favorite_vs_underdog(df)
    week_stats = analyze_by_week(df)
    confidence_points_by_week = analyze_confidence_points_by_week(df)
    division_stats = analyze_division_games(df)
    primetime_stats = analyze_primetime_games(df)
    thursday_stats = analyze_thursday_games(df)
    weather_stats = analyze_weather_impact(df)
    rest_stats = analyze_rest_differential(df)
    close_game_stats = analyze_close_games(df)
    recent_form_stats = analyze_recent_form_impact(df)
    epa_mismatch_stats = analyze_epa_mismatches(df)
    qb_change_stats = analyze_qb_changes(df)
    margin_magnitude_stats = analyze_by_margin_magnitude(df)
    power_rank_stats = analyze_by_power_ranking_diff(df)
    adj_rank_stats = analyze_by_adj_overall_rank_diff(df)
    predicted_team_stats = analyze_by_predicted_team(df)
    # New top-10 model features
    power_rank_l3_stats = analyze_by_power_ranking_diff_l3(df)
    mov_diff_stats = analyze_by_avg_margin_of_victory_diff(df)
    mov_diff_l3_stats = analyze_by_avg_margin_of_victory_diff_l3(df)
    weekly_pt_diff_stats = analyze_by_avg_weekly_point_diff(df)
    spread_perf_stats = analyze_by_spread_performance_diff(df)
    sacks_diff_stats = analyze_by_sacks_suffered_diff(df)

    # Additional outlier metrics
    completion_pct_stats = analyze_by_completion_pct_diff(df)
    ypa_stats = analyze_by_yards_per_attempt_diff(df)
    ypc_stats = analyze_by_yards_per_carry_diff(df)
    turnover_stats = analyze_by_turnover_diff(df)
    spread_diff_stats = analyze_by_spread_diff(df)
    rushing_yards_stats = analyze_by_rushing_yards_diff(df)
    def_rushing_yards_stats = analyze_by_def_rushing_yards_diff(df)
    passing_yards_stats = analyze_by_passing_yards_diff(df)

    wrong_high_conf, right_low_conf, upset_wins = find_outliers(df)

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NFL Prediction Analysis Report</title>
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
            .stat-box {{
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-card h3 {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .stat-card .value {{
                font-size: 36px;
                font-weight: bold;
                margin: 10px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background-color: #013369;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .accuracy-good {{
                color: #28a745;
                font-weight: bold;
            }}
            .accuracy-bad {{
                color: #dc3545;
                font-weight: bold;
            }}
            .accuracy-medium {{
                color: #ffc107;
                font-weight: bold;
            }}
            .timestamp {{
                color: #666;
                font-size: 12px;
                text-align: right;
                margin-top: 20px;
            }}
            .insight {{
                background: #e7f3ff;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 15px 0;
            }}
            .warning {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
            }}
            .danger {{
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 15px 0;
            }}
            .section-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>🏈 NFL Prediction Model Analysis Report</h1>
        
        <div class="stat-grid">
            <div class="stat-card">
                <h3>Total Predictions</h3>
                <div class="value">{overall['total']}</div>
            </div>
            <div class="stat-card">
                <h3>Correct</h3>
                <div class="value">{overall['correct']}</div>
            </div>
            <div class="stat-card">
                <h3>Incorrect</h3>
                <div class="value">{overall['incorrect']}</div>
            </div>
            <div class="stat-card">
                <h3>Accuracy</h3>
                <div class="value">{overall['accuracy']:.1f}%</div>
            </div>
        </div>
        
        <h2>🚨 Key Insights for Future Betting</h2>
        <div class="section-grid">
            <div class="stat-box">
                <h3>Division Games</h3>
                <table>
                    <tr>
                        <th>Game Type</th>
                        <th>Accuracy</th>
                    </tr>
                    {''.join([f"<tr><td>{row['game_type']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in division_stats.iterrows()])}
                </table>
                {f"<div class='warning'><strong>⚠️ Warning:</strong> Model struggles with division games ({division_stats[division_stats['game_type'] == 'Division Game']['accuracy'].values[0]:.1f}% accuracy). Consider lowering confidence on these matchups.</div>" if len(division_stats[division_stats['game_type'] == 'Division Game']) > 0 and division_stats[division_stats['game_type'] == 'Division Game']['accuracy'].values[0] < 55 else ''}
            </div>
            
            <div class="stat-box">
                <h3>Prime Time Performance</h3>
                <table>
                    <tr>
                        <th>Game Type</th>
                        <th>Accuracy</th>
                    </tr>
                    {''.join([f"<tr><td>{row['game_type']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in primetime_stats.iterrows()])}
                </table>
                {f"<div class='danger'><strong>🚫 Avoid:</strong> Prime time games show {primetime_stats[primetime_stats['game_type'] == 'Prime Time']['accuracy'].values[0]:.1f}% accuracy. Skip or reduce stakes on SNF/MNF.</div>" if len(primetime_stats[primetime_stats['game_type'] == 'Prime Time']) > 0 and primetime_stats[primetime_stats['game_type'] == 'Prime Time']['accuracy'].values[0] < 50 else ''}
            </div>
        </div>
        
        {f"<div class='stat-box'><h3>Thursday Night Football</h3><div class='danger'><strong>🚫 Red Flag:</strong> Thursday games: {thursday_stats['correct']}/{thursday_stats['total']} ({thursday_stats['accuracy']:.1f}%). Notorious for upsets - consider skipping entirely.</div></div>" if thursday_stats and thursday_stats['accuracy'] < 50 else ''}
        
        <div class="section-grid">
            <div class="stat-box">
                <h3>Weather Conditions</h3>
                {'<table><tr><th>Condition</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + ''.join([f"<tr><td>{row['condition']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in weather_stats.iterrows()]) + '</table>' if weather_stats is not None and len(weather_stats) > 0 else '<p>Insufficient weather data available.</p>'}
            </div>
            
            <div class="stat-box">
                <h3>Rest Advantage Impact</h3>
                <table>
                    <tr>
                        <th>Rest Differential</th>
                        <th>Accuracy</th>
                    </tr>
                    {''.join([f"<tr><td>{row['rest_advantage']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in rest_stats.iterrows()])}
                </table>
            </div>
        </div>
        
        <div class="stat-box">
            <h2>🎯 Close Game Performance</h2>
            <table>
                <tr>
                    <th>Game Type</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['game_type']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in close_game_stats.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Strategy:</strong> {f"Close games (<3pt margin) have {close_game_stats[close_game_stats['game_type'].str.contains('Close')]['accuracy'].values[0]:.1f}% accuracy - essentially coin flips. Consider passing on these." if len(close_game_stats[close_game_stats['game_type'].str.contains('Close')]) > 0 and close_game_stats[close_game_stats['game_type'].str.contains('Close')]['accuracy'].values[0] < 55 else "Model handles close games well - trust the prediction."}
            </div>
        </div>
        
        <div class="stat-box">
            <h2>🔥 Recent Form Analysis</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['category']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in recent_form_stats.iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>📊 EPA Mismatch Analysis</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['category']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in epa_mismatch_stats.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> When passing and rushing EPA contradict each other (one team dominant passing but weak rushing, opponent opposite), these games are harder to predict.
            </div>
        </div>
        
        {f'''<div class="stat-box">
            <h2>🏈 Quarterback Change Impact</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['category']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in qb_change_stats.iterrows()])}
            </table>
            <div class="{'warning' if qb_change_stats[(qb_change_stats['category'] == 'Home QB Changed') | (qb_change_stats['category'] == 'Away QB Changed')]['accuracy'].mean() < qb_change_stats[qb_change_stats['category'] == 'No QB Change']['accuracy'].values[0] - 5 else 'insight'}">
                <strong>{'⚠️ Warning:' if qb_change_stats[(qb_change_stats['category'] == 'Home QB Changed') | (qb_change_stats['category'] == 'Away QB Changed')]['accuracy'].mean() < qb_change_stats[qb_change_stats['category'] == 'No QB Change']['accuracy'].values[0] - 5 else '💡 Insight:'}</strong> {f"QB changes significantly impact prediction accuracy. The model performs {qb_change_stats[qb_change_stats['category'] == 'No QB Change']['accuracy'].values[0] - qb_change_stats[(qb_change_stats['category'] == 'Home QB Changed') | (qb_change_stats['category'] == 'Away QB Changed')]['accuracy'].mean():.1f}% worse when a QB changes. Exercise caution or reduce stakes on these games." if qb_change_stats[(qb_change_stats['category'] == 'Home QB Changed') | (qb_change_stats['category'] == 'Away QB Changed')]['accuracy'].mean() < qb_change_stats[qb_change_stats['category'] == 'No QB Change']['accuracy'].values[0] - 5 else "QB changes don't significantly impact model accuracy. The model handles QB transitions well."}
            </div>
        </div>''' if qb_change_stats is not None else ''}
        
        <div class="stat-box">
            <h2>📊 Performance by Week</h2>
            <table>
                <tr>
                    <th>Week</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>Week {row['week']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in week_stats.iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>🎯 Confidence Points by Week (Correct Picks Only)</h2>
            <table>
                <tr>
                    <th>Week</th>
                    <th>Correct Picks</th>
                    <th>Total Confidence Points</th>
                </tr>
                {''.join([f"<tr><td>Week {row['week']}</td><td>{row['correct_picks']}</td><td>{row['confidence_points']:.0f}</td></tr>" for _, row in confidence_points_by_week.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> Higher confidence points indicate winning on higher-ranked picks. This metric rewards correct predictions that the model was more confident about.
            </div>
        </div>
        
        <div class="stat-box">
            <h2>🎯 Favorite vs Underdog Picks</h2>
            <table>
                <tr>
                    <th>Pick Type</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['pick_type']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in fav_vs_dog.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> {'The model performs better picking favorites - be cautious with underdog picks.' if fav_vs_dog[fav_vs_dog['pick_type'] == 'Favorite']['accuracy'].values[0] > fav_vs_dog[fav_vs_dog['pick_type'] == 'Underdog']['accuracy'].values[0] else 'The model finds value in underdogs!' if fav_vs_dog[fav_vs_dog['pick_type'] == 'Underdog']['accuracy'].values[0] > fav_vs_dog[fav_vs_dog['pick_type'] == 'Favorite']['accuracy'].values[0] else 'Model performs equally on favorites and underdogs.'}
            </div>
        </div>
        
        <div class="stat-box">
            <h2>🏠 Home vs Away Favorite</h2>
            <table>
                <tr>
                    <th>Favorite Location</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['favorite']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in spread_location.iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>🏠 Model Picks: Home vs Away Team</h2>
            <table>
                <tr>
                    <th>Model Pick Location</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['pick_location']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in home_away_picks.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> {'The model performs better picking home teams - road wins may be harder to predict.' if home_away_picks[home_away_picks['pick_location'] == 'Home Team']['accuracy'].values[0] > home_away_picks[home_away_picks['pick_location'] == 'Away Team']['accuracy'].values[0] else 'The model performs better picking away teams - potential road game edge!' if home_away_picks[home_away_picks['pick_location'] == 'Away Team']['accuracy'].values[0] > home_away_picks[home_away_picks['pick_location'] == 'Home Team']['accuracy'].values[0] else 'Model performs equally whether picking home or away teams.'}
            </div>
        </div>
        
        <div class="stat-box">
            <h2>📏 Performance by Spread Size</h2>
            <table>
                <tr>
                    <th>Spread Range</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['spread_range']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in spread_magnitude.iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>💪 Confidence vs Accuracy</h2>
            <table>
                <tr>
                    <th>Confidence Level</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                    <th>Avg Margin</th>
                </tr>
                {''.join([f"<tr><td>{row['confidence_level']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td><td>{row['avg_margin']:.2f}</td></tr>" for _, row in confidence_perf.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> {'Higher confidence correlates with better accuracy - the model is well-calibrated!' if confidence_perf.iloc[-1]['accuracy'] > confidence_perf.iloc[0]['accuracy'] else 'Confidence does not strongly correlate with accuracy - model calibration may need improvement.'}
            </div>
        </div>
        
        <div class="stat-box">
            <h2>📊 Performance by Margin Magnitude</h2>
            <table>
                <tr>
                    <th>Predicted Margin</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['margin_range']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in margin_magnitude_stats.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> Shows how well the model predicts based on the expected margin of victory. Low margins typically indicate close games (harder to predict), while high margins suggest dominant matchups.
            </div>
        </div>
        
        <h2 style="color: #013369; border-bottom: 3px solid #D50A0A; padding-bottom: 10px; margin-top: 40px;">⭐ TOP 10 MODEL FEATURES ANALYSIS</h2>
        <div class="insight" style="margin-bottom: 30px;">
            <strong>📊 About This Section:</strong> These features have the highest importance in the model's decision-making process, combining for ~51% of predictions. Understanding how the model performs with different values of these features helps identify potential outliers and high-confidence picks.
        </div>
        
        {'<div class="section-grid"><div class="stat-box"><h2>⭐ #1: Power Ranking Diff L3 (9.06% importance)</h2><table><tr><th>Recent Power Rank Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['power_rank_diff_l3']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in power_rank_l3_stats.iterrows()]) + '</table><div class="warning"><strong>⭐ MOST IMPORTANT FEATURE:</strong> Recent power ranking differential (last 3 games) is the single most influential factor in predictions. Pay close attention to this metric.</div></div></div>' if len(power_rank_l3_stats) > 0 else '<div class="warning"><strong>⚠️ Data Not Available:</strong> Power Ranking L3 data not yet tracked. This is the #1 most important feature (9.06%). Update predict.py to include power_ranking_diff_l3.</div>'}
        
        <div class="section-grid">
            <div class="stat-box">
                <h2>#2: Power Ranking Differential (7.29%)</h2>
                <table>
                    <tr>
                        <th>Power Rank Diff</th>
                        <th>Correct</th>
                        <th>Total</th>
                        <th>Accuracy</th>
                    </tr>
                    {''.join([f"<tr><td>{row['power_rank_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in power_rank_stats.iterrows()])}
                </table>
                <div class="insight">
                    <strong>💡 Insight:</strong> Overall season power ranking differential. Larger differences indicate bigger mismatches.
                </div>
            </div>
            
            {'<div class="stat-box"><h2>#3: Avg Margin of Victory Diff (6.49%)</h2><table><tr><th>MOV Differential</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['mov_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in mov_diff_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Season-long margin of victory differential. Shows which team has been winning more decisively.</div></div>' if len(mov_diff_stats) > 0 else '<div class="stat-box"><h2>#3: Avg Margin of Victory Diff (6.49%)</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Update predict.py to include avg_margin_of_victory_diff.</div></div>'}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>#4: Avg MOV Diff L3 (4.84%)</h2><table><tr><th>Recent MOV Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['mov_diff_l3']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in mov_diff_l3_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Recent 3-game margin of victory differential. Captures current form.</div></div>' if len(mov_diff_l3_stats) > 0 else '<div class="stat-box"><h2>#4: Avg MOV Diff L3 (4.84%)</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Update predict.py to include avg_margin_of_victory_diff_l3.</div></div>'}
            
            {'<div class="stat-box"><h2>⚡ #5: Avg Weekly Point Diff L3 (OPTIMIZED +7.7% → 24.3% Variance)</h2><table><tr><th>Weekly Point Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['weekly_pt_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in weekly_pt_diff_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> <span style="color: #FF6B35; font-weight: bold;">BUCKET-OPTIMIZED</span> - 3-game rolling average with data-driven boundaries.</div></div>' if len(weekly_pt_diff_stats) > 0 else '<div class="stat-box"><h2>#5: Avg Weekly Point Diff (4.47%)</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Update predict.py to include avg_weekly_point_diff_l3.</div></div>'}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>⚡ #7: Spread Performance Diff (OPTIMIZED +23.8% → 23.8% Variance 🔥)</h2><table><tr><th>ATS Record Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['spread_perf_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in spread_perf_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> <span style="color: #D50A0A; font-weight: bold;">CRITICAL FIX</span> - Previous bins had 0% variance! Now properly detects ATS record outliers.</div></div>' if len(spread_perf_stats) > 0 else '<div class="stat-box"><h2>#7: Spread Performance Diff (4.11%)</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Update predict.py to include spread_performance_diff.</div></div>'}
            
            {'<div class="stat-box"><h2>#10: Sacks Suffered Diff (3.54%)</h2><table><tr><th>Defensive Pressure</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['sacks_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in sacks_diff_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Sacks suffered differential. Measures pass protection and QB pressure.</div></div>' if len(sacks_diff_stats) > 0 else '<div class="stat-box"><h2>#10: Sacks Suffered Diff (3.54%)</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Update predict.py to include sacks_suffered_avg_diff.</div></div>'}
        </div>
        
        <h2 style="color: #013369; margin-top: 40px;">📊 Additional Outlier Detection Metrics</h2>
        <p style="margin-bottom: 20px;">These metrics help identify potential prediction outliers based on game conditions and team performance differentials.</p>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>Completion % Differential</h2><table><tr><th>Completion % Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['completion_pct_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in completion_pct_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Passing efficiency differential. Large disparities in completion percentage can indicate QB/receiver quality gaps.</div></div>' if len(completion_pct_stats) > 0 else '<div class="stat-box"><h2>Completion % Differential</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Add completion_pct_diff to training_data table.</div></div>'}
            
            {'<div class="stat-box"><h2>⚡ Yards Per Attempt Differential (OPTIMIZED +8.6% → 19% Variance)</h2><table><tr><th>YPA Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['yards_per_attempt_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in ypa_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> <span style="color: #FF6B35; font-weight: bold;">BUCKET-OPTIMIZED</span> - Passing explosiveness with 6 data-driven buckets for better outlier detection.</div></div>' if len(ypa_stats) > 0 else '<div class="stat-box"><h2>Yards Per Attempt Differential</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Add yards_per_attempt_diff to training_data table.</div></div>'}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>⚡ Yards Per Carry Differential (OPTIMIZED +3.5% → 30% Variance)</h2><table><tr><th>YPC Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['yards_per_carry_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in ypc_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> <span style="color: #FF6B35; font-weight: bold;">BUCKET-OPTIMIZED</span> - Rushing efficiency differential with data-driven bucket boundaries for maximum outlier detection.</div></div>' if len(ypc_stats) > 0 else '<div class="stat-box"><h2>Yards Per Carry Differential</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Add yards_per_carry_diff to training_data table.</div></div>'}
            
            {'<div class="stat-box"><h2>Turnover Differential</h2><table><tr><th>Turnover Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['turnover_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in turnover_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Net turnovers (forced - committed). Turnovers are one of the most predictive factors in football.</div></div>' if len(turnover_stats) > 0 else '<div class="stat-box"><h2>Turnover Differential</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Turnover differential is calculated from individual turnover columns.</div></div>'}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>Spread Differential (Sportsbook Agreement)</h2><table><tr><th>Spread Difference</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['spread_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in spread_diff_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Model performance based on how much different sportsbooks disagree on the spread. Larger differences may indicate uncertainty.</div></div>' if len(spread_diff_stats) > 0 else '<div class="stat-box"><h2>Spread Differential (Sportsbook Agreement)</h2><div class="warning"><strong>⚠️ Data Not Available:</strong> Add spread_diff to predictions table.</div></div>'}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>🏃 Rushing Yards Differential (⚠️ TOP OUTLIER - 30% Variance)</h2><table><tr><th>Rushing Yards Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['rushing_yards_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in rushing_yards_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> <span style="color: #D50A0A; font-weight: bold;">HIGHEST VARIANCE METRIC</span> - Teams with major rushing advantages show very different model performance. Ground game dominance is highly predictive.</div></div>' if len(rushing_yards_stats) > 0 else ''}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>🛡️ Defensive Rushing Yards Differential</h2><table><tr><th>Def Rushing Yards Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['def_rushing_yards_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in def_rushing_yards_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Run defense differential (negative = better defense). Elite run defenses create predictable outcomes.</div></div>' if len(def_rushing_yards_stats) > 0 else ''}
        </div>
        
        <div class="section-grid">
            {'<div class="stat-box"><h2>✈️ Passing Yards Differential</h2><table><tr><th>Passing Yards Diff</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>' + "".join([f"<tr><td>{row['passing_yards_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in passing_yards_stats.iterrows()]) + '</table><div class="insight"><strong>💡 Insight:</strong> Passing game advantage/disadvantage. Major differentials indicate aerial dominance or struggles.</div></div>' if len(passing_yards_stats) > 0 else ''}
        </div>
        
        <div class="section-grid">
            <div class="stat-box">
                <h2>📊 Performance by Adjusted Overall Rank Differential</h2>
                <table>
                    <tr>
                        <th>Adj Rank Diff</th>
                        <th>Correct</th>
                        <th>Total</th>
                        <th>Accuracy</th>
                    </tr>
                    {''.join([f"<tr><td>{row['rank_diff']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in adj_rank_stats.iterrows()])}
                </table>
                <div class="insight">
                    <strong>💡 Insight:</strong> Similar to power rankings but adjusted for schedule strength and recent performance. Higher differences = clearer favorites.
                </div>
            </div>
        </div>
        
        <div class="stat-box">
            <h2>🏆 Best Teams When Model Picks Them</h2>
            <p>Teams with the highest accuracy when the model predicts them to win:</p>
            <table>
                <tr>
                    <th>Team</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['team']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in predicted_team_stats.head(10).iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Insight:</strong> These are the most reliable teams to bet on when the model picks them. High accuracy means the model understands their strengths well.
            </div>
        </div>
        
        <div class="stat-box">
            <h2>⚠️ Worst Teams When Model Picks Them</h2>
            <p>Teams with the lowest accuracy when the model predicts them to win:</p>
            <table>
                <tr>
                    <th>Team</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['team']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in predicted_team_stats.tail(10).iterrows()])}
            </table>
            <div class="warning">
                <strong>⚠️ Caution:</strong> When the model picks these teams, it often gets it wrong. These teams may have unpredictable performance or the model may not understand their dynamics well. Consider fading the model or reducing stakes when these teams are predicted winners.
            </div>
        </div>
        
        <div class="stat-box">
            <h2>🏆 Best Teams (Overall)</h2>
            <table>
                <tr>
                    <th>Team</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['team']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in team_stats.head(10).iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>📉 Worst Teams (Overall)</h2>
            <table>
                <tr>
                    <th>Team</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
                {''.join([f"<tr><td>{row['team']}</td><td>{row['correct']}</td><td>{row['total']}</td><td class='{get_accuracy_class(row['accuracy'])}'>{row['accuracy']:.1f}%</td></tr>" for _, row in team_stats.tail(10).iterrows()])}
            </table>
            <div class="warning">
                <strong>⚠️ Caution:</strong> These teams show unpredictable patterns. Consider avoiding or reducing stakes on games involving these teams.
            </div>
        </div>
        
        <div class="stat-box">
            <h2>❌ High Confidence Misses</h2>
            <p>Games where the model was very confident but wrong - learn from these mistakes:</p>
            <table>
                <tr>
                    <th>Week</th>
                    <th>Matchup</th>
                    <th>Predicted</th>
                    <th>Spread</th>
                    <th>Confidence</th>
                    <th>Margin</th>
                </tr>
                {''.join([f"<tr><td>{row['week']}</td><td>{row['away_team']} @ {row['home_team']}</td><td>{row['predicted_winner']}</td><td>{row['spread']}</td><td>{row['confidence']}</td><td>{row['cover_spread_by']:.2f}</td></tr>" for _, row in wrong_high_conf.iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>✅ Low Confidence Hits</h2>
            <p>Games where the model had low confidence but was correct - potential value plays:</p>
            <table>
                <tr>
                    <th>Week</th>
                    <th>Matchup</th>
                    <th>Predicted</th>
                    <th>Spread</th>
                    <th>Confidence</th>
                    <th>Margin</th>
                </tr>
                {''.join([f"<tr><td>{row['week']}</td><td>{row['away_team']} @ {row['home_team']}</td><td>{row['predicted_winner']}</td><td>{row['spread']}</td><td>{row['confidence']}</td><td>{row['cover_spread_by']:.2f}</td></tr>" for _, row in right_low_conf.iterrows()])}
            </table>
        </div>
        
        <div class="stat-box">
            <h2>🎲 Biggest Underdog Wins</h2>
            <p>Large spread underdogs (+7 or more) that the model correctly picked:</p>
            <table>
                <tr>
                    <th>Week</th>
                    <th>Matchup</th>
                    <th>Underdog</th>
                    <th>Spread</th>
                    <th>Confidence</th>
                </tr>
                {''.join([f"<tr><td>{row['week']}</td><td>{row['away_team']} @ {row['home_team']}</td><td>{row['predicted_winner']}</td><td>+{row['spread']}</td><td>{row['confidence']}</td></tr>" for _, row in upset_wins.iterrows()])}
            </table>
            <div class="insight">
                <strong>💡 Value Plays:</strong> These are high-value wins where the model found edge against large spreads. Look for similar patterns in future games.
            </div>
        </div>
        
        <div class="timestamp">
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {output_file}")


def save_accuracy_metrics_to_db(df):
    """Save all accuracy metrics to database for lookup during prediction."""
    engine = get_db_engine()

    # Create a table to store all accuracy metrics
    create_table_query = """
    CREATE TABLE IF NOT EXISTS prediction_accuracy_metrics (
        metric_type TEXT NOT NULL,
        category TEXT NOT NULL,
        subcategory TEXT,
        correct INTEGER NOT NULL,
        total INTEGER NOT NULL,
        accuracy REAL NOT NULL,
        additional_data TEXT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (metric_type, category, subcategory)
    )
    """

    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        conn.commit()

        # Clear existing data
        conn.execute(text("DELETE FROM prediction_accuracy_metrics"))
        conn.commit()

    # Prepare all metrics for insertion
    metrics_data = []

    # 1. Spread magnitude
    spread_magnitude = analyze_by_spread_magnitude(df)
    for _, row in spread_magnitude.iterrows():
        metrics_data.append(
            {
                "metric_type": "spread_magnitude",
                "category": str(row["spread_range"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 2. Margin magnitude
    margin_magnitude = analyze_by_margin_magnitude(df)
    for _, row in margin_magnitude.iterrows():
        metrics_data.append(
            {
                "metric_type": "margin_magnitude",
                "category": str(row["margin_range"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 3. Power ranking differential
    power_rank_stats = analyze_by_power_ranking_diff(df)
    for _, row in power_rank_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "power_ranking_diff",
                "category": str(row["power_rank_diff"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 4. Adjusted overall rank differential
    adj_rank_stats = analyze_by_adj_overall_rank_diff(df)
    for _, row in adj_rank_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "adj_overall_rank_diff",
                "category": str(row["rank_diff"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 5. Predicted team performance
    predicted_team_stats = analyze_by_predicted_team(df)
    for _, row in predicted_team_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "predicted_team",
                "category": str(row["team"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 6. Home vs Away picks
    home_away_picks = analyze_home_away_picks(df)
    for _, row in home_away_picks.iterrows():
        metrics_data.append(
            {
                "metric_type": "pick_location",
                "category": str(row["pick_location"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 7. Favorite vs Underdog
    fav_vs_dog = analyze_favorite_vs_underdog(df)
    for _, row in fav_vs_dog.iterrows():
        metrics_data.append(
            {
                "metric_type": "favorite_underdog",
                "category": str(row["pick_type"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 8. Division games
    division_stats = analyze_division_games(df)
    for _, row in division_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "division_game",
                "category": str(row["game_type"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 9. Prime time games
    primetime_stats = analyze_primetime_games(df)
    for _, row in primetime_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "primetime",
                "category": str(row["game_type"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 10. Rest differential
    rest_stats = analyze_rest_differential(df)
    for _, row in rest_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "rest_differential",
                "category": str(row["rest_advantage"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 11. Close games
    close_game_stats = analyze_close_games(df)
    for _, row in close_game_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "close_game",
                "category": str(row["game_type"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 12. Recent form impact
    recent_form_stats = analyze_recent_form_impact(df)
    for _, row in recent_form_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "recent_form",
                "category": str(row["category"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 13. EPA mismatches
    epa_mismatch_stats = analyze_epa_mismatches(df)
    for _, row in epa_mismatch_stats.iterrows():
        metrics_data.append(
            {
                "metric_type": "epa_mismatch",
                "category": str(row["category"]),
                "subcategory": None,
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "accuracy": float(row["accuracy"]),
            }
        )

    # 14. QB changes (if available)
    qb_change_stats = analyze_qb_changes(df)
    if qb_change_stats is not None:
        for _, row in qb_change_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "qb_change",
                    "category": str(row["category"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 14b. QB change (simplified - any change vs no change)
    qb_change_any_stats = analyze_qb_change_any(df)
    if qb_change_any_stats is not None:
        for _, row in qb_change_any_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "qb_change_any",
                    "category": str(row["category"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 15. Weather conditions (if available)
    weather_stats = analyze_weather_impact(df)
    if weather_stats is not None:
        for _, row in weather_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "weather",
                    "category": str(row["condition"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 16. Power ranking diff L3 (TOP FEATURE!)
    power_rank_l3_stats = analyze_by_power_ranking_diff_l3(df)
    if len(power_rank_l3_stats) > 0:
        for _, row in power_rank_l3_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "power_ranking_diff_l3",
                    "category": str(row["power_rank_diff_l3"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 17. Average margin of victory differential
    mov_stats = analyze_by_avg_margin_of_victory_diff(df)
    if len(mov_stats) > 0:
        for _, row in mov_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "avg_margin_of_victory_diff",
                    "category": str(row["mov_diff"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 18. Average margin of victory differential L3
    mov_l3_stats = analyze_by_avg_margin_of_victory_diff_l3(df)
    if len(mov_l3_stats) > 0:
        for _, row in mov_l3_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "avg_margin_of_victory_diff_l3",
                    "category": str(row["mov_diff_l3"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 19. Average weekly point differential (overall)
    weekly_pt_stats = analyze_by_avg_weekly_point_diff(df)
    if len(weekly_pt_stats) > 0:
        for _, row in weekly_pt_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "avg_weekly_point_diff",
                    "category": str(row["weekly_pt_diff"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 20. Spread performance differential (ATS records)
    spread_perf_stats = analyze_by_spread_performance_diff(df)
    if len(spread_perf_stats) > 0:
        for _, row in spread_perf_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "spread_performance_diff",
                    "category": str(row["spread_perf_diff"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 21. Sacks suffered differential (defensive pressure)
    sacks_stats = analyze_by_sacks_suffered_diff(df)
    if len(sacks_stats) > 0:
        for _, row in sacks_stats.iterrows():
            metrics_data.append(
                {
                    "metric_type": "sacks_suffered_avg_diff",
                    "category": str(row["sacks_diff"]),
                    "subcategory": None,
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    # 22. Completion percentage differential
    completion_pct_stats = analyze_by_completion_pct_diff(df)
    if len(completion_pct_stats) > 0:
        for _, row in completion_pct_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "completion_pct_diff",
                        "category": str(row["completion_pct_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 23. Yards per attempt differential
    ypa_stats = analyze_by_yards_per_attempt_diff(df)
    if len(ypa_stats) > 0:
        for _, row in ypa_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "yards_per_attempt_diff",
                        "category": str(row["yards_per_attempt_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 24. Yards per carry differential
    ypc_stats = analyze_by_yards_per_carry_diff(df)
    if len(ypc_stats) > 0:
        for _, row in ypc_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "yards_per_carry_diff",
                        "category": str(row["yards_per_carry_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 25. Turnover differential
    turnover_stats = analyze_by_turnover_diff(df)
    if len(turnover_stats) > 0:
        for _, row in turnover_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "turnover_diff",
                        "category": str(row["turnover_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 26. Spread differential
    spread_diff_stats = analyze_by_spread_diff(df)
    if len(spread_diff_stats) > 0:
        for _, row in spread_diff_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "spread_diff",
                        "category": str(row["spread_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 27. Rushing yards differential (TOP OUTLIER METRIC - 30% variance)
    rushing_yards_stats = analyze_by_rushing_yards_diff(df)
    if len(rushing_yards_stats) > 0:
        for _, row in rushing_yards_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "rushing_yards_diff",
                        "category": str(row["rushing_yards_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 28. Defensive rushing yards differential
    def_rushing_yards_stats = analyze_by_def_rushing_yards_diff(df)
    if len(def_rushing_yards_stats) > 0:
        for _, row in def_rushing_yards_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "def_rushing_yards_diff",
                        "category": str(row["def_rushing_yards_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # 29. Passing yards differential
    passing_yards_stats = analyze_by_passing_yards_diff(df)
    if len(passing_yards_stats) > 0:
        for _, row in passing_yards_stats.iterrows():
            if pd.notna(row["accuracy"]) and row["total"] > 0:
                metrics_data.append(
                    {
                        "metric_type": "passing_yards_diff",
                        "category": str(row["passing_yards_diff"]),
                        "subcategory": None,
                        "correct": int(row["correct"]),
                        "total": int(row["total"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )

    # Convert to DataFrame and insert
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_sql(
        "prediction_accuracy_metrics", engine, if_exists="append", index=False
    )

    print(f"\n✅ Saved {len(metrics_data)} accuracy metrics to database")
    print(f"   Metric types: {metrics_df['metric_type'].nunique()}")
    print(f"   Total categories: {len(metrics_data)}")

    # Print summary by metric type
    print("\n📊 Metrics saved by type:")
    for metric_type in sorted(metrics_df["metric_type"].unique()):
        count = len(metrics_df[metrics_df["metric_type"] == metric_type])
        print(f"   - {metric_type}: {count} categories")


def lookup_accuracy_metric(metric_type, value):
    """
    Look up accuracy for a given metric type and value.

    Args:
        metric_type: Type of metric (e.g., 'spread_magnitude', 'predicted_team', 'adj_overall_rank_diff')
        value: The actual value to look up (e.g., 8 for spread, 'BAL' for team, 20 for rank diff)

    Returns:
        dict with accuracy info or None if not found

    Examples:
        lookup_accuracy_metric('spread_magnitude', 8)
        lookup_accuracy_metric('predicted_team', 'BAL')
        lookup_accuracy_metric('adj_overall_rank_diff', 20)
    """
    engine = get_db_engine()

    # Determine which bucket the value falls into based on metric type
    if metric_type == "spread_magnitude":
        # OPTIMIZED bins for 35.3% variance (was 18.2%)
        if value <= 2.5:
            category = "0-2.5 (Very Close)"
        elif value <= 3.0:
            category = "2.5-3.0 (Close)"
        elif value <= 4.0:
            category = "3.0-4.0 (Tight)"
        elif value <= 6.5:
            category = "4.0-6.5 (Medium)"
        elif value <= 7.5:
            category = "6.5-7.5 (Large)"
        else:
            category = "7.5+ (Blowout)"

    elif metric_type == "margin_magnitude":
        # Same bins as spread
        if value <= 3:
            category = "0-3 (Close)"
        elif value <= 7:
            category = "3-7 (Medium)"
        elif value <= 14:
            category = "7-14 (Large)"
        else:
            category = "14+ (Blowout)"

    elif metric_type == "power_ranking_diff":
        # Bins: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6+
        if value <= 0.2:
            category = "0-0.2 (Low)"
        elif value <= 0.4:
            category = "0.2-0.4 (Medium)"
        elif value <= 0.6:
            category = "0.4-0.6 (High)"
        else:
            category = "0.6+ (Very High)"

    elif metric_type == "adj_overall_rank_diff":
        # Bins: 0-8, 8-16, 16-24, 24+
        if value <= 8:
            category = "0-8 (Low)"
        elif value <= 16:
            category = "8-16 (Medium)"
        elif value <= 24:
            category = "16-24 (High)"
        else:
            category = "24+ (Very High)"

    elif metric_type == "predicted_team":
        # Direct lookup - value is the team name
        category = str(value)

    elif metric_type == "pick_location":
        # value should be 'Home Team' or 'Away Team'
        category = str(value)

    elif metric_type == "favorite_underdog":
        # value should be 'Favorite' or 'Underdog'
        category = str(value)

    elif metric_type == "division_game":
        # value should be 'Division Game' or 'Non-Division'
        category = str(value)

    elif metric_type == "primetime":
        # value should be 'Prime Time' or 'Regular'
        category = str(value)

    elif metric_type == "close_game":
        # value should be 'Close (<3 pts)' or 'Not Close (≥3 pts)'
        category = str(value)

    elif metric_type == "weather":
        # Direct lookup - value is the condition name
        category = str(value)

    elif metric_type == "qb_change":
        # Direct lookup - value is the category name
        category = str(value)

    elif metric_type == "qb_change_any":
        # Direct lookup - value should be 'No QB Change' or 'QB Change (Any)'
        category = str(value)

    elif metric_type == "rest_differential":
        # Direct lookup - value is the rest advantage category
        category = str(value)

    elif metric_type == "recent_form":
        # Direct lookup - value is the category
        category = str(value)

    elif metric_type == "epa_mismatch":
        # Direct lookup - value is the category
        category = str(value)

    elif metric_type == "power_ranking_diff_l3":
        # Bins: 0-0.2, 0.2-0.4, 0.4+
        if value <= 0.2:
            category = "0-0.2 (Low)"
        elif value <= 0.4:
            category = "0.2-0.4 (Medium)"
        else:
            category = "0.4+ (High)"

    elif metric_type == "avg_margin_of_victory_diff":
        # Bins: 0-3, 3-7, 7-14, 14+
        if value <= 3:
            category = "0-3 (Close)"
        elif value <= 7:
            category = "3-7 (Medium)"
        elif value <= 14:
            category = "7-14 (Large)"
        else:
            category = "14+ (Dominant)"

    elif metric_type == "avg_margin_of_victory_diff_l3":
        # Same bins as avg_margin_of_victory_diff
        if value <= 3:
            category = "0-3 (Close)"
        elif value <= 7:
            category = "3-7 (Medium)"
        elif value <= 14:
            category = "7-14 (Large)"
        else:
            category = "14+ (Dominant)"

    elif metric_type == "avg_weekly_point_diff":
        # OPTIMIZED bins for 24.3% variance (was 16.7%) - using L3 metric
        if value <= 5.667:
            category = "0-5.7 (Even)"
        elif value <= 9.333:
            category = "5.7-9.3 (Moderate)"
        else:
            category = "9.3+ (Strong)"

    elif metric_type == "spread_performance_diff":
        # OPTIMIZED bins for 23.8% variance (was 0%!)
        if value < -7.0:
            category = "< -7.0 (Very Poor ATS)"
        elif value < -3.5:
            category = "-7.0 to -3.5 (Poor ATS)"
        elif value < -0.559:
            category = "-3.5 to -0.6 (Below Avg)"
        elif value < 1.722:
            category = "-0.6 to 1.7 (Avg)"
        elif value < 4.969:
            category = "1.7 to 5.0 (Good ATS)"
        else:
            category = "> 5.0 (Elite ATS)"

    elif metric_type == "sacks_suffered_avg_diff":
        # Bins: 0-0.5, 0.5-1.0, 1.0-1.5, 1.5+
        if value <= 0.5:
            category = "0-0.5 (Even)"
        elif value <= 1.0:
            category = "0.5-1.0 (Moderate)"
        elif value <= 1.5:
            category = "1.0-1.5 (Large)"
        else:
            category = "1.5+ (Extreme)"

    elif metric_type == "completion_pct_diff":
        # Bins: < -10%, -10% to -5%, -5% to 5%, 5% to 10%, > 10%
        if value <= 0.05:
            category = "-5% to 5%"
        elif value <= 0.10:
            category = "5% to 10%"
        else:
            category = "> 10%"

    elif metric_type == "yards_per_attempt_diff":
        # OPTIMIZED bins for 19.0% variance (was 10.4%)
        if value < -1.542:
            category = "< -1.54 (Very Weak)"
        elif value < -0.744:
            category = "-1.54 to -0.74 (Weak)"
        elif value < -0.013:
            category = "-0.74 to 0 (Below Avg)"
        elif value < 0.501:
            category = "0 to 0.50 (Above Avg)"
        elif value < 1.341:
            category = "0.50 to 1.34 (Strong)"
        else:
            category = "> 1.34 (Elite)"

    elif metric_type == "yards_per_carry_diff":
        # OPTIMIZED bins for 30.0% variance (was 26.5%)
        if value < -0.747:
            category = "< -0.75 (Very Weak)"
        elif value < -0.163:
            category = "-0.75 to -0.16 (Weak)"
        elif value < 0.351:
            category = "-0.16 to 0.35 (Even)"
        elif value < 0.869:
            category = "0.35 to 0.87 (Strong)"
        else:
            category = "> 0.87 (Dominant)"

    elif metric_type == "turnover_diff":
        # Bins: <= -2, -1, 0 (Even), +1, +2, >= +3
        if value <= -2:
            category = "<= -2"
        elif value == -1:
            category = "-1"
        elif value == 0:
            category = "0 (Even)"
        elif value == 1:
            category = "+1"
        elif value == 2:
            category = "+2"
        else:
            category = ">= +3"

    elif metric_type == "spread_diff":
        # Bins: 0, 0.5-2, 2.5-4, >4 (using absolute value)
        abs_value = abs(value)
        if abs_value <= 0.25:
            category = "0"
        elif abs_value <= 2.25:
            category = "0.5-2"
        elif abs_value <= 4:
            category = "2.5-4"
        else:
            category = ">4"

    elif metric_type == "rushing_yards_diff":
        # Bins: < -100, -100 to -25, -25 to 25, 25 to 100, > 100
        if value < -100:
            category = "< -100 (Very Weak)"
        elif value <= -25:
            category = "-100 to -25 (Weak)"
        elif value <= 25:
            category = "-25 to 25 (Even)"
        elif value <= 100:
            category = "25 to 100 (Strong)"
        else:
            category = "> 100 (Dominant)"

    elif metric_type == "def_rushing_yards_diff":
        # Bins: < -150, -150 to -50, -50 to 50, 50 to 150, > 150
        if value < -150:
            category = "< -150 (Elite D)"
        elif value <= -50:
            category = "-150 to -50 (Good D)"
        elif value <= 50:
            category = "-50 to 50 (Average)"
        elif value <= 150:
            category = "50 to 150 (Poor D)"
        else:
            category = "> 150 (Very Poor D)"

    elif metric_type == "passing_yards_diff":
        # Bins: < -200, -200 to -75, -75 to 75, 75 to 200, > 200
        if value < -200:
            category = "< -200 (Very Weak)"
        elif value <= -75:
            category = "-200 to -75 (Weak)"
        elif value <= 75:
            category = "-75 to 75 (Even)"
        elif value <= 200:
            category = "75 to 200 (Strong)"
        else:
            category = "> 200 (Dominant)"

    else:
        return None

    # Query the database
    query = text(
        """
        SELECT metric_type, category, correct, total, accuracy, last_updated
        FROM prediction_accuracy_metrics
        WHERE metric_type = :metric_type AND category = :category
    """
    )

    with engine.connect() as conn:
        result = conn.execute(
            query, {"metric_type": metric_type, "category": category}
        ).fetchone()

    if result:
        return {
            "metric_type": result[0],
            "category": result[1],
            "correct": result[2],
            "total": result[3],
            "accuracy": result[4],
            "last_updated": result[5],
        }
    else:
        return None


def get_all_metrics_for_prediction(prediction_data):
    """
    Get all relevant accuracy metrics for a prediction.

    Args:
        prediction_data: dict containing prediction details:
            - spread: float
            - margin: float (predicted margin)
            - power_ranking_diff: float
            - adj_overall_rank_diff: float
            - predicted_winner: str (team abbreviation)
            - predicted_home: bool (whether picking home team)
            - is_favorite: bool (whether picking favorite)
            - is_division_game: bool
            - is_primetime: bool
            - (optional) other fields

    Returns:
        dict with all applicable accuracy metrics
    """
    metrics = {}

    # Spread magnitude
    if "spread" in prediction_data:
        result = lookup_accuracy_metric(
            "spread_magnitude", abs(prediction_data["spread"])
        )
        if result:
            metrics["spread_magnitude"] = result

    # Margin magnitude
    if "cover_spread_by" in prediction_data:
        result = lookup_accuracy_metric(
            "margin_magnitude", abs(prediction_data["cover_spread_by"])
        )
        if result:
            metrics["margin_magnitude"] = result

    # Power ranking diff
    if "power_ranking_diff" in prediction_data:
        result = lookup_accuracy_metric(
            "power_ranking_diff", abs(prediction_data["power_ranking_diff"])
        )
        if result:
            metrics["power_ranking_diff"] = result

    # Adjusted overall rank diff
    if "adj_overall_rank_diff" in prediction_data:
        result = lookup_accuracy_metric(
            "adj_overall_rank_diff", abs(prediction_data["adj_overall_rank_diff"])
        )
        if result:
            metrics["adj_overall_rank_diff"] = result

    # Predicted team
    if "predicted_winner" in prediction_data:
        result = lookup_accuracy_metric(
            "predicted_team", prediction_data["predicted_winner"]
        )
        if result:
            metrics["predicted_team"] = result

    # Pick location
    if "predicted_home" in prediction_data:
        location = "Home Team" if prediction_data["predicted_home"] else "Away Team"
        result = lookup_accuracy_metric("pick_location", location)
        if result:
            metrics["pick_location"] = result

    # Favorite vs Underdog
    if "is_favorite" in prediction_data:
        pick_type = "Favorite" if prediction_data["is_favorite"] else "Underdog"
        result = lookup_accuracy_metric("favorite_underdog", pick_type)
        if result:
            metrics["favorite_underdog"] = result

    # Division game
    if "is_division_game" in prediction_data:
        game_type = (
            "Division Game" if prediction_data["is_division_game"] else "Non-Division"
        )
        result = lookup_accuracy_metric("division_game", game_type)
        if result:
            metrics["division_game"] = result

    # Primetime
    if "is_primetime" in prediction_data:
        game_type = "Prime Time" if prediction_data["is_primetime"] else "Regular"
        result = lookup_accuracy_metric("primetime", game_type)
        if result:
            metrics["primetime"] = result

    return metrics


def print_prediction_metrics(prediction_data, show_warnings=True):
    """
    Print all accuracy metrics for a prediction with color-coded warnings.

    Args:
        prediction_data: dict with prediction details
        show_warnings: bool, whether to show warning messages for low accuracy
    """
    metrics = get_all_metrics_for_prediction(prediction_data)

    if not metrics:
        print("⚠️  No accuracy metrics found for this prediction")
        return

    print("\n" + "=" * 80)
    print("📊 PREDICTION ACCURACY LOOKUP")
    print("=" * 80)

    warnings = []

    for metric_type, data in metrics.items():
        accuracy = data["accuracy"]

        # Color code based on accuracy
        if accuracy >= 60:
            emoji = "✅"
            color = "GOOD"
        elif accuracy >= 50:
            emoji = "⚠️ "
            color = "MEDIUM"
        else:
            emoji = "❌"
            color = "BAD"
            warnings.append(f"{metric_type}: {data['category']} ({accuracy:.1f}%)")

        print(f"\n{emoji} {metric_type.upper().replace('_', ' ')}")
        print(f"   Category: {data['category']}")
        print(
            f"   Historical Accuracy: {accuracy:.1f}% ({data['correct']}/{data['total']})"
        )
        print(f"   Status: {color}")

    if show_warnings and warnings:
        print("\n" + "=" * 80)
        print("🚨 WARNING: Low Accuracy Indicators")
        print("=" * 80)
        for warning in warnings:
            print(f"   ⚠️  {warning}")
        print("\n   Consider reducing stake or skipping this prediction.")

    print("\n" + "=" * 80)
