import pandas as pd

ALL_BASE_FEATURES = [
    "rest_diff",
    "div_game",
    "avg_margin_of_victory_diff",
    "avg_margin_of_victory_diff_l3",
    "spread_performance_diff",
    "spread_performance_diff_l3",
    "completion_pct_diff",
    "yards_per_attempt_diff",
    "yards_per_carry_diff",
    "passing_epa_diff",
    "rushing_epa_diff",
    "passing_yards_diff",
    "passing_tds_diff",
    "passing_interceptions_diff",
    "sacks_suffered_diff",
    "sack_fumbles_lost_diff",
    "rushing_yards_diff",
    "rushing_tds_diff",
    "rushing_fumbles_lost_diff",
    "def_passing_yards_diff",
    "def_passing_tds_diff",
    "def_passing_interceptions_diff",
    "def_sacks_suffered_diff",
    "def_sack_fumbles_lost_diff",
    "def_rushing_yards_diff",
    "def_rushing_tds_diff",
    "def_rushing_fumbles_lost_diff",
    "power_ranking_diff",
    "power_ranking_diff_l3",
    "win_pct_diff",
    "win_pct_diff_l3",
    "sos_diff",
    "adj_offensive_rank_diff",
    "adj_defensive_rank_diff",
    "adj_overall_rank_diff",
    "avg_weekly_point_diff",
    "avg_weekly_point_diff_l3",
    "passing_yards_avg_diff",
    "passing_tds_avg_diff",
    "passing_interceptions_avg_diff",
    "sacks_suffered_avg_diff",
    "sack_fumbles_lost_avg_diff",
    "rushing_yards_avg_diff",
    "rushing_tds_avg_diff",
    "rushing_fumbles_lost_avg_diff",
    "def_passing_yards_avg_diff",
    "def_passing_tds_avg_diff",
    "def_passing_interceptions_avg_diff",
    "def_sacks_suffered_avg_diff",
    "def_sack_fumbles_lost_avg_diff",
    "def_rushing_yards_avg_diff",
    "def_rushing_tds_avg_diff",
    "def_rushing_fumbles_lost_avg_diff",
]

FEATURE_SELECTION_BASE_FEATURES = [
    "spread_performance_diff",
    "power_ranking_diff_l3",
    "rushing_epa_diff",
    "spread_performance_diff_l3",
    "passing_epa_diff",
    "sacks_suffered_diff",
    "avg_margin_of_victory_diff_l3",
    "avg_weekly_point_diff_l3",
    "completion_pct_diff",
    "def_passing_interceptions_avg_diff",
    "avg_margin_of_victory_diff",
    "avg_weekly_point_diff",
    "yards_per_carry_diff",
    "power_ranking_diff",
    "rushing_tds_avg_diff",
    "sacks_suffered_avg_diff",
    "passing_interceptions_avg_diff",
    "sack_fumbles_lost_diff",
    "def_passing_tds_diff",
    "def_rushing_fumbles_lost_diff",
    "sack_fumbles_lost_avg_diff",
    "rest_diff",
    "def_rushing_tds_diff",
    "sos_diff",
    "def_passing_yards_diff",
]

TARGET = "point_differential"


def engineer_features(
    df: pd.DataFrame, base_features: list = FEATURE_SELECTION_BASE_FEATURES
) -> tuple[pd.DataFrame, list]:

    if "home_score" in df.columns and "away_score" in df.columns:
        df["point_differential"] = df["home_score"] - df["away_score"]

    features = base_features
    return df, features
