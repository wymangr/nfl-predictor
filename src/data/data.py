## nflreadr Status: https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html

import nflreadpy as nfl
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sqlalchemy import text

from src.helpers.database_helpers import get_db_engine
from src.data.yahoo_spreads import YahooSpreadClient


def add_rolling_3week_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling 3-week averages and previous rolling 3-week averages for power_ranking and win_pct.
    Handles week 1/2 by using previous season's last weeks if available, else fills with 0.5 (neutral).
    """
    # Sort for correct rolling
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    # Calculate rolling 3-game average for power_ranking (crosses seasons)
    df["power_ranking_l3"] = df.groupby("team")["power_ranking"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    # Calculate rolling 3-game average for win_pct (crosses seasons)
    df["win_pct_l3"] = df.groupby("team")["win_pct"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    # Shift to get previous week's 3-game averages (don't include current week)
    df["prev_power_ranking_l3"] = df.groupby("team")["power_ranking_l3"].shift(1)
    df["prev_win_pct_l3"] = df.groupby("team")["win_pct_l3"].shift(1)
    # For first game of first season, fill with 0.5
    df["prev_power_ranking_l3"] = df["prev_power_ranking_l3"].fillna(0.5)
    df["prev_win_pct_l3"] = df["prev_win_pct_l3"].fillna(0.5)
    return df


def get_sos(seasons=None):
    """
    Calculate team power rankings and strength of schedule (SOS) for each team by season and week.
    Power rankings are based on win%, offensive stats, defensive stats, and turnovers.
    SOS is calculated based on the composite power ranking of opponents from the PREVIOUS week.
    Week 1 will have SOS = 0.5 for all teams (no prior week stats available).
    Only calculates for weeks where games have actually been completed (have scores).
    Handles bye weeks automatically by using accumulated stats.
    Results are stored in the 'team_power_rankings' table.

    Args:
        seasons: List of seasons to calculate SOS for. If None, calculates for all seasons.
    """
    engine = get_db_engine()

    # Get only completed games (with scores)
    if seasons:
        season_filter = f"AND season IN ({','.join(map(str, seasons))})"
    else:
        season_filter = ""

    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    games_df = pd.read_sql_query(
        text(
            f"""
            SELECT * FROM games
            WHERE (
                (season != {current_season} {season_filter})
                OR (season = {current_season} AND week <= {current_week})
            )
            AND ((season != {current_season} and game_type = 'REG') OR (season = {current_season}))
            """
        ),
        engine,
    )


    if len(games_df) == 0:
        print("No completed games found")
        return pd.DataFrame()

    # Get accumulated team stats (cumulative through each week)
    team_stats_accumulated = pd.read_sql_query(
        text(
            f"""
        SELECT * FROM team_stats_accumulated
        {('WHERE season IN (' + ','.join(map(str, seasons)) + ')') if seasons else ''}
    """
        ),
        engine,
    )

    # Calculate win percentage for each team at each week
    home_games = games_df[
        ["season", "week", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    home_games["team"] = home_games["home_team"]
    home_games["win"] = (home_games["home_score"] > home_games["away_score"]).astype(
        int
    )

    away_games = games_df[
        ["season", "week", "away_team", "home_team", "away_score", "home_score"]
    ].copy()
    away_games["team"] = away_games["away_team"]
    away_games["win"] = (away_games["away_score"] > away_games["home_score"]).astype(
        int
    )

    all_games = (
        pd.concat(
            [
                home_games[["season", "week", "team", "win"]],
                away_games[["season", "week", "team", "win"]],
            ]
        )
        .sort_values(["season", "team", "week"])
        .reset_index(drop=True)
    )

    # Calculate cumulative wins and win percentage
    all_games["cumulative_wins"] = all_games.groupby(["season", "team"])["win"].cumsum()
    all_games["games_played"] = all_games.groupby(["season", "team"]).cumcount() + 1
    all_games["win_pct"] = all_games["cumulative_wins"] / all_games["games_played"]

    # Aggregate to get one record per team-week with win%
    team_wins = (
        all_games.groupby(["season", "team", "week"])
        .agg({"cumulative_wins": "last", "games_played": "last", "win_pct": "last"})
        .reset_index()
    )

    # Merge win percentage with accumulated stats
    team_performance = pd.merge(
        team_stats_accumulated, team_wins, on=["season", "team", "week"], how="left"
    )

    # Forward fill win_pct and games_played for bye weeks (not replace with 0!)
    # This ensures teams keep their previous week's record during bye weeks
    team_performance["win_pct"] = (
        team_performance.groupby(["season", "team"])["win_pct"].ffill().fillna(0)
    )
    team_performance["cumulative_wins"] = (
        team_performance.groupby(["season", "team"])["cumulative_wins"]
        .ffill()
        .fillna(0)
    )
    team_performance["games_played"] = (
        team_performance.groupby(["season", "team"])["games_played"].ffill().fillna(0)
    )

    # SHIFT DATA: For ranking week N, use stats from week N-1
    # This means week 1 uses previous season's final week stats
    team_performance = team_performance.sort_values(
        ["team", "season", "week"]
    ).reset_index(drop=True)

    # Shift all stat columns by 1 row within each team (lagged stats)
    stat_columns = [
        "win_pct",
        "cumulative_wins",
        "games_played",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "sacks_suffered",
        "sack_fumbles_lost",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
        "def_passing_yards",
        "def_passing_tds",
        "def_passing_interceptions",
        "def_sacks_suffered",
        "def_sack_fumbles_lost",
        "def_rushing_yards",
        "def_rushing_tds",
        "def_rushing_fumbles_lost",
    ]

    for col in stat_columns:
        if col in team_performance.columns:
            team_performance[col] = team_performance.groupby("team")[col].shift(1)

    # Fill NaN (week 1 of first season) with 0 or carry from previous season
    team_performance[stat_columns] = team_performance[stat_columns].fillna(0)

    # Calculate per-game stats (to normalize for different numbers of games played)
    team_performance["games_played_safe"] = team_performance["games_played"].replace(
        0, 1
    )  # Avoid division by zero

    # Offensive metrics (per game)
    team_performance["off_yards_per_game"] = (
        team_performance["passing_yards"] + team_performance["rushing_yards"]
    ) / team_performance["games_played_safe"]
    team_performance["off_tds_per_game"] = (
        team_performance["passing_tds"] + team_performance["rushing_tds"]
    ) / team_performance["games_played_safe"]
    team_performance["off_turnovers_per_game"] = (
        team_performance["passing_interceptions"]
        + team_performance["sack_fumbles_lost"]
        + team_performance["rushing_fumbles_lost"]
    ) / team_performance["games_played_safe"]

    # Defensive metrics (per game) - lower is better, so we'll invert these
    team_performance["def_yards_per_game"] = (
        team_performance["def_passing_yards"] + team_performance["def_rushing_yards"]
    ) / team_performance["games_played_safe"]
    team_performance["def_tds_per_game"] = (
        team_performance["def_passing_tds"] + team_performance["def_rushing_tds"]
    ) / team_performance["games_played_safe"]
    team_performance["def_turnovers_per_game"] = (
        team_performance["def_passing_interceptions"]
        + team_performance["def_sack_fumbles_lost"]
        + team_performance["def_rushing_fumbles_lost"]
    ) / team_performance["games_played_safe"]

    # Calculate power ranking for each team at each week
    # We'll rank teams within each season-week combination

    # Group by season-week to calculate rankings
    ranking_results = []

    for (season, week), group in team_performance.groupby(["season", "week"]):
        n_teams = len(group)
        if n_teams <= 1:
            group["power_ranking"] = 0.5
            group["offensive_rank"] = 1
            group["defensive_rank"] = 1
            group["overall_rank"] = 1
            ranking_results.append(group)
            continue

        # For metrics where higher is better
        group["win_pct_rank"] = group["win_pct"].rank(pct=True)
        group["off_yards_rank"] = group["off_yards_per_game"].rank(pct=True)
        group["off_tds_rank"] = group["off_tds_per_game"].rank(pct=True)
        group["def_turnovers_rank"] = group["def_turnovers_per_game"].rank(pct=True)

        # For metrics where lower is better (invert)
        group["off_turnovers_rank"] = 1 - group["off_turnovers_per_game"].rank(pct=True)
        group["def_yards_rank"] = 1 - group["def_yards_per_game"].rank(pct=True)
        group["def_tds_rank"] = 1 - group["def_tds_per_game"].rank(pct=True)

        # Calculate offensive composite score (0-1 scale)
        group["offensive_score"] = (
            group["off_yards_rank"] * 0.50
            + group["off_tds_rank"] * 0.40
            + group["off_turnovers_rank"] * 0.10
        )

        # Calculate defensive composite score (0-1 scale)
        group["defensive_score"] = (
            group["def_yards_rank"] * 0.50
            + group["def_tds_rank"] * 0.35
            + group["def_turnovers_rank"] * 0.15
        )

        # Composite power ranking (weighted average)
        group["power_ranking"] = (
            group["win_pct_rank"] * 0.35
            + group["off_yards_rank"] * 0.15
            + group["off_tds_rank"] * 0.12
            + group["off_turnovers_rank"] * 0.08
            + group["def_yards_rank"] * 0.15
            + group["def_tds_rank"] * 0.10
            + group["def_turnovers_rank"] * 0.05
        )

        # Add actual ranks (1-32 style, not percentiles)
        # Fill NaN with a high rank (worst) before converting to int
        group["offensive_rank"] = (
            group["offensive_score"]
            .rank(ascending=False, method="min")
            .fillna(32)
            .astype(int)
        )
        group["defensive_rank"] = (
            group["defensive_score"]
            .rank(ascending=False, method="min")
            .fillna(32)
            .astype(int)
        )
        group["overall_rank"] = (
            group["power_ranking"]
            .rank(ascending=False, method="min")
            .fillna(32)
            .astype(int)
        )

        ranking_results.append(group)

    team_performance = pd.concat(ranking_results, ignore_index=True)

    # Create lookup for team performance (include scores for later SOS adjustment)
    team_performance_lookup = team_performance[
        [
            "season",
            "team",
            "week",
            "power_ranking",
            "win_pct",
            "games_played",
            "offensive_rank",
            "defensive_rank",
            "overall_rank",
            "offensive_score",
            "defensive_score",
        ]
    ].copy()

    # Now calculate SOS using power rankings
    sos_records = []

    # Get list of completed weeks per season
    completed_weeks = (
        games_df.groupby("season")["week"].apply(lambda x: sorted(x.unique())).to_dict()
    )

    # Recreate all_games with opponent info for SOS calculation
    home_games = games_df[["season", "week", "home_team", "away_team"]].copy()
    home_games["team"] = home_games["home_team"]
    home_games["opponent"] = home_games["away_team"]

    away_games = games_df[["season", "week", "away_team", "home_team"]].copy()
    away_games["team"] = away_games["away_team"]
    away_games["opponent"] = away_games["home_team"]

    all_games_with_opp = (
        pd.concat(
            [
                home_games[["season", "week", "team", "opponent"]],
                away_games[["season", "week", "team", "opponent"]],
            ]
        )
        .sort_values(["season", "team", "week"])
        .reset_index(drop=True)
    )

    for season in sorted(completed_weeks.keys()):
        weeks = completed_weeks[season]
        teams = all_games_with_opp[all_games_with_opp["season"] == season][
            "team"
        ].unique()

        for team in teams:
            for week in weeks:
                if week == 1:
                    # Week 1: No prior stats, SOS = 0.5 (neutral)
                    sos_records.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "sos": 0.5,
                            "opponents_faced": 0,
                        }
                    )
                else:
                    # Get all opponents this team has faced before this week
                    team_games_before_week = all_games_with_opp[
                        (all_games_with_opp["season"] == season)
                        & (all_games_with_opp["team"] == team)
                        & (all_games_with_opp["week"] < week)
                    ].copy()

                    if len(team_games_before_week) == 0:
                        # Team hasn't played yet (bye weeks at start, rare)
                        sos_records.append(
                            {
                                "season": season,
                                "week": week,
                                "team": team,
                                "sos": 0.5,
                                "opponents_faced": 0,
                            }
                        )
                        continue

                    # For each opponent faced, get their power ranking from the week BEFORE they played
                    opponent_rankings = []

                    for _, game in team_games_before_week.iterrows():
                        opponent = game["opponent"]
                        game_week = game["week"]

                        if game_week == 1:
                            # Week 1 games - no prior ranking, use 0.5 (neutral)
                            opponent_rankings.append(0.5)
                        else:
                            # Look up opponent's most recent power ranking before game_week
                            opp_record = team_performance_lookup[
                                (team_performance_lookup["season"] == season)
                                & (team_performance_lookup["team"] == opponent)
                                & (team_performance_lookup["week"] < game_week)
                            ]

                            if len(opp_record) > 0:
                                # Use the most recent power ranking
                                latest_record = opp_record.iloc[-1]
                                opponent_rankings.append(latest_record["power_ranking"])
                            else:
                                # Opponent hasn't played yet (shouldn't happen)
                                opponent_rankings.append(0.5)

                    # Calculate average opponent power ranking (SOS)
                    if opponent_rankings:
                        avg_opp_ranking = sum(opponent_rankings) / len(
                            opponent_rankings
                        )
                    else:
                        avg_opp_ranking = 0.5

                    sos_records.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "sos": avg_opp_ranking,
                            "opponents_faced": len(team_games_before_week),
                        }
                    )

    # Create DataFrame for SOS values
    sos_df = pd.DataFrame(sos_records)

    # Merge SOS with team performance to create adjusted rankings
    team_performance_with_sos = pd.merge(
        team_performance_lookup,
        sos_df[["season", "week", "team", "sos"]],
        on=["season", "week", "team"],
        how="left",
    )

    # Calculate SOS-adjusted ranks for each season-week
    adjusted_ranking_results = []

    for (season, week), group in team_performance_with_sos.groupby(["season", "week"]):
        if len(group) <= 1 or week == 1:
            # Not enough teams or first week - no adjustment
            group["adj_offensive_rank"] = group["offensive_rank"]
            group["adj_defensive_rank"] = group["defensive_rank"]
            group["adj_overall_rank"] = group["overall_rank"]
            # Drop the score columns for week 1 too
            group_result = group[
                [
                    "season",
                    "team",
                    "week",
                    "power_ranking",
                    "win_pct",
                    "games_played",
                    "offensive_rank",
                    "defensive_rank",
                    "overall_rank",
                    "sos",
                    "adj_offensive_rank",
                    "adj_defensive_rank",
                    "adj_overall_rank",
                ]
            ]
            adjusted_ranking_results.append(group_result)
            continue

        # Merge with offensive/defensive scores from team_performance_lookup (already has them)
        group_with_scores = group.copy()

        # SOS adjustment factor:
        # - Higher SOS (harder schedule) = boost the score
        # - Lower SOS (easier schedule) = reduce the score
        # Normalize SOS around mean, then apply scale factor for meaningful rank changes
        sos_mean = group_with_scores["sos"].mean()
        sos_adjustment = (group_with_scores["sos"] - sos_mean) * 1.5  # Scale factor

        # Apply SOS adjustment to scores
        group_with_scores["adj_offensive_score"] = (
            group_with_scores["offensive_score"] + sos_adjustment
        )
        group_with_scores["adj_defensive_score"] = (
            group_with_scores["defensive_score"] + sos_adjustment
        )
        group_with_scores["adj_power_score"] = (
            group_with_scores["power_ranking"] + sos_adjustment
        )

        # Re-rank based on adjusted scores
        group_with_scores["adj_offensive_rank"] = (
            group_with_scores["adj_offensive_score"]
            .rank(ascending=False, method="min")
            .fillna(32)
            .astype(int)
        )
        group_with_scores["adj_defensive_rank"] = (
            group_with_scores["adj_defensive_score"]
            .rank(ascending=False, method="min")
            .fillna(32)
            .astype(int)
        )
        group_with_scores["adj_overall_rank"] = (
            group_with_scores["adj_power_score"]
            .rank(ascending=False, method="min")
            .fillna(32)
            .astype(int)
        )

        # Keep only the columns we need (drop offensive_score and defensive_score)
        group_result = group_with_scores[
            [
                "season",
                "team",
                "week",
                "power_ranking",
                "win_pct",
                "games_played",
                "offensive_rank",
                "defensive_rank",
                "overall_rank",
                "sos",
                "adj_offensive_rank",
                "adj_defensive_rank",
                "adj_overall_rank",
            ]
        ]
        adjusted_ranking_results.append(group_result)

    team_performance_final = pd.concat(adjusted_ranking_results, ignore_index=True)

    # Calculate rolling 3-game averages for power_ranking and win_pct
    print("Calculating power ranking and win % rolling averages (last 3 games)...")

    team_performance_final = add_rolling_3week_averages(team_performance_final)

    # Sort by team and time
    # team_performance_final = team_performance_final.sort_values(
    #     ["team", "season", "week"]
    # )

    # # Calculate rolling 3-game average for power_ranking (crosses seasons)
    # team_performance_final["power_ranking_l3"] = team_performance_final.groupby("team")[
    #     "power_ranking"
    # ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # # Calculate rolling 3-game average for win_pct (crosses seasons)
    # team_performance_final["win_pct_l3"] = team_performance_final.groupby("team")[
    #     "win_pct"
    # ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # # Shift to get previous week's 3-game averages (don't include current week)
    # team_performance_final["prev_power_ranking_l3"] = team_performance_final.groupby(
    #     "team"
    # )["power_ranking_l3"].shift(1)
    # team_performance_final["prev_win_pct_l3"] = team_performance_final.groupby("team")[
    #     "win_pct_l3"
    # ].shift(1)

    # # For first game of first season, fill with 0 or neutral values
    # team_performance_final["prev_power_ranking_l3"] = team_performance_final[
    #     "prev_power_ranking_l3"
    # ].fillna(0.5)
    # team_performance_final["prev_win_pct_l3"] = team_performance_final[
    #     "prev_win_pct_l3"
    # ].fillna(0.5)

    # Save the adjusted power rankings with rolling averages
    team_performance_final.to_sql(
        "team_power_rankings", engine, if_exists="replace", index=False
    )

    print(
        f"Team power rankings calculated for {len(team_performance_final)} team-week combinations"
    )
    return team_performance_final


def get_previous_week_key(row, team_stats_accumulated):
    current_season = row["season"]
    current_week = row["week"]

    if current_week > 1:
        return current_season, current_week - 1
    else:
        # Find the max week in the *previous* season
        previous_season = current_season - 1
        # Check if previous season data exists in accumulated stats
        if previous_season in team_stats_accumulated["season"].unique():
            max_week_prev_season = team_stats_accumulated[
                team_stats_accumulated["season"] == previous_season
            ]["week"].max()
            return previous_season, max_week_prev_season
        else:
            return None, None  # Mark to be dropped


def merge_most_recent_lookup(
    training_data, lookup_df, lookup_cols, home_prefix, away_prefix
):
    """
    Merge lookup data using most recent available week for each team.
    ALWAYS uses data from PREVIOUS weeks only to avoid data leakage.
    For week 1 games, uses previous season's final week data.

    Args:
        training_data: DataFrame with games
        lookup_df: DataFrame with lookup data (must have columns: season, week, team, + lookup_cols)
        lookup_cols: List of column names to merge from lookup_df
        home_prefix: Prefix for home team columns (e.g., 'home_')
        away_prefix: Prefix for away team columns (e.g., 'away_')

    Returns:
        Updated training_data DataFrame with merged columns
    """
    lookup_sorted = lookup_df.sort_values(["season", "team", "week"])

    result_rows = []
    for idx, game in training_data.iterrows():
        season = game["season"]
        week = game["week"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        # ALWAYS use data from PREVIOUS weeks only to avoid data leakage
        # This ensures that week N predictions use only week N-1 (and earlier) data
        if week == 1:
            # For Week 1, use previous season's data
            season_filter = season - 1
            week_filter = 100  # Use any week from previous season
        else:
            season_filter = season
            week_filter = week - 1  # Only use data from weeks BEFORE this one

        # Find most recent home team data
        home_data = lookup_sorted[
            (lookup_sorted["team"] == home_team)
            & (
                (
                    (lookup_sorted["season"] == season_filter)
                    & (lookup_sorted["week"] <= week_filter)
                )
                | (lookup_sorted["season"] < season_filter)
            )
        ]

        # Find most recent away team data
        away_data = lookup_sorted[
            (lookup_sorted["team"] == away_team)
            & (
                (
                    (lookup_sorted["season"] == season_filter)
                    & (lookup_sorted["week"] <= week_filter)
                )
                | (lookup_sorted["season"] < season_filter)
            )
        ]

        # Add data if available
        if len(home_data) > 0:
            home_rec = home_data.iloc[-1]
            for col in lookup_cols:
                game[f"{home_prefix}{col}"] = home_rec[col]
        else:
            for col in lookup_cols:
                game[f"{home_prefix}{col}"] = None

        if len(away_data) > 0:
            away_rec = away_data.iloc[-1]
            for col in lookup_cols:
                game[f"{away_prefix}{col}"] = away_rec[col]
        else:
            for col in lookup_cols:
                game[f"{away_prefix}{col}"] = None

        result_rows.append(game)

    return pd.DataFrame(result_rows)


def backfil_data(backfil_season: int = 2003):
    current_year = nfl.get_current_season()
    seasons = list(range(backfil_season, current_year + 1))
    YahooSpreadClient().get_all_week_spreads()

    # Load team stats
    team_stats = nfl.load_team_stats(seasons=seasons, summary_level="week")
    team_stats_df = team_stats.to_pandas()
    # team_stats_df = team_stats_df[team_stats_df["season_type"] == "REG"]
    team_stats_df = team_stats_df[
        (team_stats_df["season_type"] == "REG")
        | (team_stats_df["season"] == current_year)
    ].copy()

    ## DEBUG - ADD THIS TO CUT OF DATA TO TEST PREVIOUS WEEKS DATA
    # team_stats_df = team_stats_df[~((team_stats_df['season'] == 2025) & (team_stats_df['week'] == 15))]
    ##

    # Define desired stats - only use columns that actually exist in the data
    # Basic offense stats that have defensive equivalents
    basic_offense_stats = [
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "sacks_suffered",
        "sack_fumbles_lost",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
    ]
    # Additional offense stats without defensive equivalents (EPA, efficiency metrics)
    additional_offense_stats = [
        col
        for col in ["completions", "attempts", "carries", "passing_epa", "rushing_epa"]
        if col in team_stats_df.columns
    ]
    offense_stats = basic_offense_stats + additional_offense_stats
    defense_stats = [
        "def_passing_yards",
        "def_passing_tds",
        "def_passing_interceptions",
        "def_sacks_suffered",
        "def_sack_fumbles_lost",
        "def_rushing_yards",
        "def_rushing_tds",
        "def_rushing_fumbles_lost",
    ]
    desired_columns = ["season", "week", "team"] + offense_stats + defense_stats

    # Only use basic offense stats for opponent mapping (those with defensive equivalents)
    opponent_stats_df = team_stats_df[
        ["season", "week", "team"] + basic_offense_stats
    ].copy()
    opponent_stats_df = opponent_stats_df.rename(
        columns={
            "team": "opponent_team",
            "passing_yards": "def_passing_yards",
            "passing_tds": "def_passing_tds",
            "passing_interceptions": "def_passing_interceptions",
            "sacks_suffered": "def_sacks_suffered",
            "sack_fumbles_lost": "def_sack_fumbles_lost",
            "rushing_yards": "def_rushing_yards",
            "rushing_tds": "def_rushing_tds",
            "rushing_fumbles_lost": "def_rushing_fumbles_lost",
        }
    )

    team_stats_df = pd.merge(
        team_stats_df,
        opponent_stats_df,
        how="left",
        on=["season", "week", "opponent_team"],
    )

    # Get cumulative stats making sure to fill in bye weeks with previous weeks data
    stats_to_accumulate = offense_stats + defense_stats
    # Only select columns that exist in team_stats_df
    columns_to_select = ["season", "week", "team"] + offense_stats + defense_stats
    df_subset = team_stats_df[columns_to_select].copy()
    df_subset[stats_to_accumulate] = df_subset.groupby(["season", "team"])[
        stats_to_accumulate
    ].cumsum()

    all_seasons = df_subset["season"].unique()
    all_teams = df_subset["team"].unique()
    max_week_per_season = df_subset.groupby("season")["week"].max().to_dict()

    multi_index = pd.MultiIndex.from_product(
        [all_seasons, all_teams, range(1, 20)], names=["season", "team", "week"]
    )
    full_index_df = pd.DataFrame(index=multi_index).reset_index()

    ### I CHANGED THIS ###
    # full_index_df = full_index_df[
    #     full_index_df.apply(
    #         lambda row: row["week"] <= max_week_per_season.get(row["season"], 0), axis=1
    #     )
    # ]

    ### THIS ADDS CURRENT WEEK OF DATA INTO SOS ###
    # full_index_df = full_index_df[
    #     full_index_df.apply(
    #         lambda row: row["week"] <= max_week_per_season.get(row["season"], 0) + 1, axis=1
    #     )
    # ]

    ### FIX??? ###
    # Determine the max scheduled week for the current season
    schedule_df = nfl.load_schedules(seasons=[current_year]).to_pandas()
    if not schedule_df.empty:
        max_scheduled_week = schedule_df["week"].max()
    else:
        max_scheduled_week = max_week_per_season.get(current_year, 0)

    def week_limit(row):
        season = row["season"]
        week = row["week"]
        max_week = max_week_per_season.get(season, 0)
        if season == current_year:
            # Only add +1 if the max week in the index is less than the scheduled max week
            if max_week < max_scheduled_week:
                return week <= max_week + 1
            else:
                return week <= max_week
        else:
            return week <= max_week

    full_index_df = full_index_df[full_index_df.apply(week_limit, axis=1)]

    df_subset_reindexed = pd.merge(
        full_index_df, df_subset, on=["season", "team", "week"], how="left"
    )

    df_subset_reindexed[stats_to_accumulate] = df_subset_reindexed.groupby(
        ["season", "team"]
    )[stats_to_accumulate].ffill()

    df_subset_reindexed.sort_values(["season", "team", "week"], inplace=True)
    team_stats_accumulated = df_subset_reindexed

    # Get game data
    games = nfl.load_schedules(seasons=seasons)
    games_df = games.to_pandas()
    games_df = games_df[
        (games_df["game_type"] == "REG") | (games_df["season"] == current_year)
    ].copy()

    yahoo_spreads_df = pd.read_sql_table("yahoo_spreads", get_db_engine())
    games_df = pd.merge(
        games_df,
        yahoo_spreads_df,
        how="left",
        left_on=["season", "week", "home_team", "away_team"],
        right_on=["season", "week", "home_team", "away_team"],
    )

    # Save team_stats, team_stats_accumulated, and games to database FIRST
    # This is required before calling get_sos() which reads from these tables
    print("Saving team stats and games to database...")
    engine = get_db_engine()
    seasons_str = ",".join(map(str, seasons))

    # Handle team_stats_accumulated table with schema migration check
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='team_stats_accumulated'"
            )
        )
        accumulated_table_exists = result.fetchone() is not None

    if accumulated_table_exists:
        # Check if new columns exist (completions, attempts, carries, passing_epa, rushing_epa)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(team_stats_accumulated)"))
            accumulated_columns = [row[1] for row in result.fetchall()]

        if (
            "completions" not in accumulated_columns
            or "attempts" not in accumulated_columns
            or "carries" not in accumulated_columns
            or "passing_epa" not in accumulated_columns
            or "rushing_epa" not in accumulated_columns
        ):
            print(
                "⚠️  Schema change detected in team_stats_accumulated - recreating table with new columns..."
            )
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS team_stats_accumulated"))
                conn.commit()
            team_stats_accumulated.to_sql(
                "team_stats_accumulated", engine, if_exists="replace", index=False
            )
        else:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        f"DELETE FROM team_stats_accumulated WHERE season IN ({seasons_str})"
                    )
                )
                conn.commit()
            team_stats_accumulated.to_sql(
                "team_stats_accumulated", engine, if_exists="append", index=False
            )
    else:
        team_stats_accumulated.to_sql(
            "team_stats_accumulated", engine, if_exists="replace", index=False
        )

    # Handle team_stats and games tables
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"DELETE FROM team_stats WHERE season IN ({seasons_str})")
            )
            conn.commit()
    except Exception:
        # Table doesn't exist yet
        pass
    try:
        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM games WHERE season IN ({seasons_str})"))
            conn.commit()
    except Exception:
        # Table doesn't exist yet
        pass

    team_stats_df.to_sql("team_stats", engine, if_exists="append", index=False)
    games_df.to_sql("games", engine, if_exists="append", index=False)

    # NOW calculate and store team power rankings with SOS (reads from database)
    print("Calculating team power rankings and strength of schedule...")
    team_performance = get_sos(seasons=seasons)

    # Drop games_played since it will be duplicated in the merge
    if "games_played" in team_performance.columns:
        team_performance.drop(columns=["games_played"], inplace=True)

    # Build training data by merging game data with accumulated team stats
    print("Building training data by merging game data with accumulated team stats...")
    desired_game_cols = [
        "game_id",
        "season",
        "game_type",
        "week",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "spread_line",
        "yahoo_spread",
        "away_rest",
        "home_rest",
        "div_game",
    ]
    training_data = games_df[desired_game_cols].copy()

    # Calculate rest_diff (positive = home team more rested)
    training_data["rest_diff"] = training_data["home_rest"] - training_data["away_rest"]

    stats_prep = team_stats_accumulated.copy()
    stats_prep = stats_prep.rename(
        columns={"season": "stats_season", "week": "stats_week"}
    )

    training_data[["stats_season", "stats_week"]] = training_data.apply(
        lambda row: pd.Series(get_previous_week_key(row, team_stats_accumulated)),
        axis=1,
    )
    training_data.dropna(subset=["stats_season", "stats_week"], inplace=True)
    training_data["stats_season"] = training_data["stats_season"].astype(int)
    training_data["stats_week"] = training_data["stats_week"].astype(int)

    # Merge home team stats
    training_data = pd.merge(
        training_data,
        stats_prep.add_prefix("home_"),
        how="left",
        left_on=["stats_season", "stats_week", "home_team"],
        right_on=["home_stats_season", "home_stats_week", "home_team"],
    )

    # Merge away team stats
    training_data = pd.merge(
        training_data,
        stats_prep.add_prefix("away_"),
        how="left",
        left_on=["stats_season", "stats_week", "away_team"],
        right_on=["away_stats_season", "away_stats_week", "away_team"],
    )

    # Merge power rankings using most recent available data
    # This handles both historical games (exact week match) and future games (use most recent week)
    print("Merging power rankings (using most recent available data for each team)...")

    # Prepare team_performance for efficient lookup
    team_performance_sorted = team_performance.sort_values(["season", "team", "week"])

    # For each game, find the most recent power rankings available (week < game week)
    # ALWAYS use data from PREVIOUS weeks only to avoid data leakage
    result_rows = []

    for idx, game in training_data.iterrows():
        season = game["season"]
        week = game["week"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        # ALWAYS use data from PREVIOUS weeks only to avoid data leakage
        # This ensures that week N predictions use only week N-1 (and earlier) data
        if week == 1:
            # For Week 1, use previous season's data
            season_filter = season - 1
            week_filter = 100  # Use any week from previous season
        else:
            season_filter = season
            week_filter = week - 1  # Only use data from weeks BEFORE this one

        # Find most recent home team rankings
        home_rankings = team_performance_sorted[
            (team_performance_sorted["team"] == home_team)
            & (
                (
                    (team_performance_sorted["season"] == season_filter)
                    & (team_performance_sorted["week"] <= week_filter)
                )
                | (team_performance_sorted["season"] < season_filter)
            )
        ]

        # Find most recent away team rankings
        away_rankings = team_performance_sorted[
            (team_performance_sorted["team"] == away_team)
            & (
                (
                    (team_performance_sorted["season"] == season_filter)
                    & (team_performance_sorted["week"] <= week_filter)
                )
                | (team_performance_sorted["season"] < season_filter)
            )
        ]

        # Add rankings if available
        if len(home_rankings) > 0 and len(away_rankings) > 0:
            home_rank = home_rankings.iloc[-1]  # Most recent
            away_rank = away_rankings.iloc[-1]  # Most recent

            # Add home team rankings
            game["home_power_ranking"] = home_rank["power_ranking"]
            game["home_win_pct"] = home_rank["win_pct"]
            game["home_sos"] = home_rank["sos"]
            game["home_adj_offensive_rank"] = home_rank["adj_offensive_rank"]
            game["home_adj_defensive_rank"] = home_rank["adj_defensive_rank"]
            game["home_adj_overall_rank"] = home_rank["adj_overall_rank"]
            game["home_prev_power_ranking_l3"] = home_rank["prev_power_ranking_l3"]
            game["home_prev_win_pct_l3"] = home_rank["prev_win_pct_l3"]

            # Add away team rankings
            game["away_power_ranking"] = away_rank["power_ranking"]
            game["away_win_pct"] = away_rank["win_pct"]
            game["away_sos"] = away_rank["sos"]
            game["away_adj_offensive_rank"] = away_rank["adj_offensive_rank"]
            game["away_adj_defensive_rank"] = away_rank["adj_defensive_rank"]
            game["away_adj_overall_rank"] = away_rank["adj_overall_rank"]
            game["away_prev_power_ranking_l3"] = away_rank["prev_power_ranking_l3"]
            game["away_prev_win_pct_l3"] = away_rank["prev_win_pct_l3"]

            result_rows.append(game)

    training_data = pd.DataFrame(result_rows)

    # Clean up temporary columns
    training_data.drop(
        columns=[
            "home_stats_season",
            "home_stats_week",
            "away_stats_season",
            "away_stats_week",
            "stats_season",
            "stats_week",
        ],
        inplace=True,
    )

    # Calculate efficiency metrics before calculating differentials
    # Completion percentage
    training_data["home_completion_pct"] = training_data[
        "home_completions"
    ] / training_data["home_attempts"].replace(0, 1)
    training_data["away_completion_pct"] = training_data[
        "away_completions"
    ] / training_data["away_attempts"].replace(0, 1)

    # Yards per attempt (passing)
    training_data["home_yards_per_attempt"] = training_data[
        "home_passing_yards"
    ] / training_data["home_attempts"].replace(0, 1)
    training_data["away_yards_per_attempt"] = training_data[
        "away_passing_yards"
    ] / training_data["away_attempts"].replace(0, 1)

    # Yards per carry (rushing)
    training_data["home_yards_per_carry"] = training_data[
        "home_rushing_yards"
    ] / training_data["home_carries"].replace(0, 1)
    training_data["away_yards_per_carry"] = training_data[
        "away_rushing_yards"
    ] / training_data["away_carries"].replace(0, 1)

    # Calculate differential features for team stats
    training_data["passing_yards_diff"] = (
        training_data["home_passing_yards"] - training_data["away_passing_yards"]
    )
    training_data["passing_tds_diff"] = (
        training_data["home_passing_tds"] - training_data["away_passing_tds"]
    )
    training_data["passing_interceptions_diff"] = (
        training_data["home_passing_interceptions"]
        - training_data["away_passing_interceptions"]
    )
    training_data["sacks_suffered_diff"] = (
        training_data["home_sacks_suffered"] - training_data["away_sacks_suffered"]
    )
    training_data["sack_fumbles_lost_diff"] = (
        training_data["home_sack_fumbles_lost"]
        - training_data["away_sack_fumbles_lost"]
    )
    training_data["rushing_yards_diff"] = (
        training_data["home_rushing_yards"] - training_data["away_rushing_yards"]
    )
    training_data["rushing_tds_diff"] = (
        training_data["home_rushing_tds"] - training_data["away_rushing_tds"]
    )
    training_data["rushing_fumbles_lost_diff"] = (
        training_data["home_rushing_fumbles_lost"]
        - training_data["away_rushing_fumbles_lost"]
    )
    training_data["def_passing_yards_diff"] = (
        training_data["home_def_passing_yards"]
        - training_data["away_def_passing_yards"]
    )
    training_data["def_passing_tds_diff"] = (
        training_data["home_def_passing_tds"] - training_data["away_def_passing_tds"]
    )
    training_data["def_passing_interceptions_diff"] = (
        training_data["home_def_passing_interceptions"]
        - training_data["away_def_passing_interceptions"]
    )
    training_data["def_sacks_suffered_diff"] = (
        training_data["home_def_sacks_suffered"]
        - training_data["away_def_sacks_suffered"]
    )
    training_data["def_sack_fumbles_lost_diff"] = (
        training_data["home_def_sack_fumbles_lost"]
        - training_data["away_def_sack_fumbles_lost"]
    )
    training_data["def_rushing_yards_diff"] = (
        training_data["home_def_rushing_yards"]
        - training_data["away_def_rushing_yards"]
    )
    training_data["def_rushing_tds_diff"] = (
        training_data["home_def_rushing_tds"] - training_data["away_def_rushing_tds"]
    )
    training_data["def_rushing_fumbles_lost_diff"] = (
        training_data["home_def_rushing_fumbles_lost"]
        - training_data["away_def_rushing_fumbles_lost"]
    )

    # Calculate efficiency differentials
    training_data["completion_pct_diff"] = (
        training_data["home_completion_pct"] - training_data["away_completion_pct"]
    )
    training_data["yards_per_attempt_diff"] = (
        training_data["home_yards_per_attempt"]
        - training_data["away_yards_per_attempt"]
    )
    training_data["yards_per_carry_diff"] = (
        training_data["home_yards_per_carry"] - training_data["away_yards_per_carry"]
    )
    training_data["passing_epa_diff"] = (
        training_data["home_passing_epa"] - training_data["away_passing_epa"]
    )
    training_data["rushing_epa_diff"] = (
        training_data["home_rushing_epa"] - training_data["away_rushing_epa"]
    )

    # Calculate differential features for power rankings
    training_data["power_ranking_diff"] = (
        training_data["home_power_ranking"] - training_data["away_power_ranking"]
    )
    training_data["win_pct_diff"] = (
        training_data["home_win_pct"] - training_data["away_win_pct"]
    )
    training_data["sos_diff"] = training_data["home_sos"] - training_data["away_sos"]

    # Calculate 3-game rolling average differentials for power rankings and win %
    if (
        "home_prev_power_ranking_l3" in training_data.columns
        and "away_prev_power_ranking_l3" in training_data.columns
    ):
        training_data["power_ranking_diff_l3"] = (
            training_data["home_prev_power_ranking_l3"]
            - training_data["away_prev_power_ranking_l3"]
        )
    if (
        "home_prev_win_pct_l3" in training_data.columns
        and "away_prev_win_pct_l3" in training_data.columns
    ):
        training_data["win_pct_diff_l3"] = (
            training_data["home_prev_win_pct_l3"]
            - training_data["away_prev_win_pct_l3"]
        )
    training_data["adj_offensive_rank_diff"] = (
        training_data["home_adj_offensive_rank"]
        - training_data["away_adj_offensive_rank"]
    )
    training_data["adj_defensive_rank_diff"] = (
        training_data["home_adj_defensive_rank"]
        - training_data["away_adj_defensive_rank"]
    )
    training_data["adj_overall_rank_diff"] = (
        training_data["home_adj_overall_rank"] - training_data["away_adj_overall_rank"]
    )

    # Calculate average weekly point differential
    # First, calculate cumulative points and games for each team
    # IMPORTANT: Only use COMPLETED games (with scores) to avoid data leakage
    # When incomplete games get scores, they shouldn't change historical averages
    games_for_avg = games_df[
        ["season", "week", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    games_for_avg = games_for_avg.dropna(subset=["home_score", "away_score"])

    # Home games - points scored by home team
    home_points = games_for_avg[["season", "week", "home_team", "home_score"]].copy()
    home_points.columns = ["season", "week", "team", "points_scored"]

    # Away games - points scored by away team
    away_points = games_for_avg[["season", "week", "away_team", "away_score"]].copy()
    away_points.columns = ["season", "week", "team", "points_scored"]

    # Combine all games
    all_points = pd.concat([home_points, away_points], ignore_index=True)
    all_points = all_points.sort_values(["team", "season", "week"])

    # Calculate cumulative points per team per season
    all_points["cumulative_points"] = all_points.groupby(["season", "team"])[
        "points_scored"
    ].cumsum()
    all_points["games_played"] = all_points.groupby(["season", "team"]).cumcount() + 1
    all_points["avg_points_per_game"] = (
        all_points["cumulative_points"] / all_points["games_played"]
    )

    # Get previous week's average (shift by 1 within each team-season)
    all_points = all_points.sort_values(["team", "season", "week"])
    all_points["prev_avg_points"] = all_points.groupby(["team", "season"])[
        "avg_points_per_game"
    ].shift(1)

    # For week 1, use previous season's final average
    all_points["prev_avg_points"] = all_points.groupby("team")[
        "prev_avg_points"
    ].ffill()
    all_points["prev_avg_points"] = all_points["prev_avg_points"].fillna(0)

    # Merge back to training data for home and away teams
    avg_points_lookup = all_points[["season", "week", "team", "prev_avg_points"]].copy()

    training_data = merge_most_recent_lookup(
        training_data, avg_points_lookup, ["prev_avg_points"], "home_", "away_"
    )
    training_data.rename(
        columns={
            "home_prev_avg_points": "home_avg_points",
            "away_prev_avg_points": "away_avg_points",
        },
        inplace=True,
    )

    training_data["avg_weekly_point_diff"] = (
        training_data["home_avg_points"] - training_data["away_avg_points"]
    )

    # Calculate last 3 games rolling average for weekly point differential
    print("Calculating avg weekly point differential (last 3 games)...")

    # Calculate rolling 3-game average for points scored (crosses seasons)
    all_points_l3 = all_points.copy()
    all_points_l3 = all_points_l3.sort_values(["team", "season", "week"])

    # Use rolling window of 3, min_periods=1 to handle first few games
    all_points_l3["rolling_3game_points"] = all_points_l3.groupby("team")[
        "points_scored"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Shift to get previous week's 3-game average (don't include current week)
    all_points_l3["prev_rolling_3game_points"] = all_points_l3.groupby("team")[
        "rolling_3game_points"
    ].shift(1)

    # For first game of first season, fill with 0
    all_points_l3["prev_rolling_3game_points"] = all_points_l3[
        "prev_rolling_3game_points"
    ].fillna(0)

    # Create lookup table
    points_l3_lookup = all_points_l3[
        ["season", "week", "team", "prev_rolling_3game_points"]
    ].copy()

    # Merge back to training data and calculate differential
    training_data = merge_most_recent_lookup(
        training_data,
        points_l3_lookup,
        ["prev_rolling_3game_points"],
        "home_points_l3_",
        "away_points_l3_",
    )
    training_data.rename(
        columns={
            "home_points_l3_prev_rolling_3game_points": "home_points_l3",
            "away_points_l3_prev_rolling_3game_points": "away_points_l3",
        },
        inplace=True,
    )

    # Calculate the differential (home - away)
    training_data["avg_weekly_point_diff_l3"] = (
        training_data["home_points_l3"] - training_data["away_points_l3"]
    )

    # Drop intermediate columns
    training_data.drop(columns=["home_points_l3", "away_points_l3"], inplace=True)

    # Calculate per-game averages for all stat columns
    stat_columns_for_avg = [
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "sacks_suffered",
        "sack_fumbles_lost",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
        "def_passing_yards",
        "def_passing_tds",
        "def_passing_interceptions",
        "def_sacks_suffered",
        "def_sack_fumbles_lost",
        "def_rushing_yards",
        "def_rushing_tds",
        "def_rushing_fumbles_lost",
    ]

    # Get the accumulated stats data (this includes bye weeks with forward-filled stats)
    stats_for_avg = team_stats_accumulated[
        ["season", "week", "team"] + stat_columns_for_avg
    ].copy()

    # For each team-season-week, get the actual games played count
    # We need to calculate this from actual games, not from all_points which only has played weeks
    # Create a complete games_played for all weeks (including bye weeks)
    games_for_count = games_df[["season", "week", "home_team", "away_team"]].copy()

    home_games_count = games_for_count[["season", "week", "home_team"]].copy()
    home_games_count.columns = ["season", "week", "team"]

    away_games_count = games_for_count[["season", "week", "away_team"]].copy()
    away_games_count.columns = ["season", "week", "team"]

    all_games_count = pd.concat([home_games_count, away_games_count], ignore_index=True)
    all_games_count = all_games_count.sort_values(["team", "season", "week"])

    # Count cumulative games per team per season
    all_games_count["games_played"] = (
        all_games_count.groupby(["season", "team"]).cumcount() + 1
    )

    # For bye weeks, we need to forward-fill the games_played count
    # First, get all season-team-week combinations from team_stats_accumulated
    all_weeks = team_stats_accumulated[["season", "team", "week"]].drop_duplicates()

    # Merge with games count
    all_weeks = pd.merge(
        all_weeks, all_games_count, on=["season", "team", "week"], how="left"
    )

    # Forward fill games_played for bye weeks
    all_weeks = all_weeks.sort_values(["team", "season", "week"])
    all_weeks["games_played"] = (
        all_weeks.groupby(["season", "team"])["games_played"].ffill().fillna(0)
    )

    # Now merge this with stats_for_avg
    stats_for_avg = pd.merge(
        stats_for_avg, all_weeks, on=["season", "week", "team"], how="left"
    )

    # Calculate per-game averages using actual games played
    stats_for_avg["games_played_safe"] = (
        stats_for_avg["games_played"].fillna(1).replace(0, 1)
    )
    for stat in stat_columns_for_avg:
        stats_for_avg[f"{stat}_avg"] = (
            stats_for_avg[stat] / stats_for_avg["games_played_safe"]
        )

    # Shift to get previous week's averages (same pattern as we did for points)
    stats_for_avg = stats_for_avg.sort_values(["team", "season", "week"])
    for stat in stat_columns_for_avg:
        stats_for_avg[f"prev_{stat}_avg"] = stats_for_avg.groupby(["team", "season"])[
            f"{stat}_avg"
        ].shift(1)
        # For week 1, use previous season's final average
        stats_for_avg[f"prev_{stat}_avg"] = stats_for_avg.groupby("team")[
            f"prev_{stat}_avg"
        ].ffill()
        stats_for_avg[f"prev_{stat}_avg"] = stats_for_avg[f"prev_{stat}_avg"].fillna(0)

    # Create lookup with only the columns we need
    avg_cols = ["season", "week", "team"] + [
        f"prev_{stat}_avg" for stat in stat_columns_for_avg
    ]
    stats_avg_lookup = stats_for_avg[avg_cols].copy()

    # Merge for home and away teams using most recent available data
    # Extract just the prev_*_avg column names for merge_most_recent_lookup
    prev_avg_cols = [f"prev_{stat}_avg" for stat in stat_columns_for_avg]
    training_data = merge_most_recent_lookup(
        training_data, stats_avg_lookup, prev_avg_cols, "home_", "away_"
    )

    # Calculate differential averages (home avg - away avg)
    for stat in stat_columns_for_avg:
        training_data[f"{stat}_avg_diff"] = (
            training_data[f"home_prev_{stat}_avg"]
            - training_data[f"away_prev_{stat}_avg"]
        )

    # Drop all individual home_ and away_ stat/ranking columns, but keep home_team, away_team, home_score, away_score
    cols_to_keep = [
        "game_id",
        "season",
        "game_type",
        "week",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "spread_line",
        "yahoo_spread",
    ]
    cols_to_drop = [
        col
        for col in training_data.columns
        if (col.startswith("home_") or col.startswith("away_"))
        and col not in cols_to_keep
    ]
    training_data.drop(columns=cols_to_drop, inplace=True)

    # Calculate average spread performance differential (how well teams beat/miss the spread)
    print("Calculating spread performance differential...")

    # Step 1 & 2: Calculate actual point differential and spread differential for all games
    games_spread_perf = games_df[
        [
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
        ]
    ].copy()
    games_spread_perf = games_spread_perf.dropna(
        subset=["home_score", "away_score", "spread_line"]
    )

    # Calculate spread performance from each team's perspective
    # For home team: their score - away score - spread_line
    # For away team: their score - home score + spread_line (spread_line is from home perspective)

    # Home team spread performance
    home_spread_perf = games_spread_perf[
        ["season", "week", "home_team", "home_score", "away_score", "spread_line"]
    ].copy()
    home_spread_perf["spread_diff"] = (
        home_spread_perf["home_score"] - home_spread_perf["away_score"]
    ) - home_spread_perf["spread_line"]
    home_spread_perf = home_spread_perf[["season", "week", "home_team", "spread_diff"]]
    home_spread_perf.columns = ["season", "week", "team", "spread_diff"]

    # Away team spread performance (from their perspective)
    away_spread_perf = games_spread_perf[
        ["season", "week", "away_team", "away_score", "home_score", "spread_line"]
    ].copy()
    away_spread_perf["spread_diff"] = (
        away_spread_perf["away_score"] - away_spread_perf["home_score"]
    ) + away_spread_perf["spread_line"]
    away_spread_perf = away_spread_perf[["season", "week", "away_team", "spread_diff"]]
    away_spread_perf.columns = ["season", "week", "team", "spread_diff"]

    # Combine all spread performance records
    all_spread_perf = pd.concat([home_spread_perf, away_spread_perf], ignore_index=True)
    all_spread_perf = all_spread_perf.sort_values(["team", "season", "week"])

    # Step 3: Calculate cumulative average spread performance for each team
    all_spread_perf["cumulative_spread_diff"] = all_spread_perf.groupby(
        ["season", "team"]
    )["spread_diff"].cumsum()
    all_spread_perf["games_played"] = (
        all_spread_perf.groupby(["season", "team"]).cumcount() + 1
    )
    all_spread_perf["avg_spread_perf"] = (
        all_spread_perf["cumulative_spread_diff"] / all_spread_perf["games_played"]
    )

    # Shift to get previous week's average (don't include current week)
    all_spread_perf = all_spread_perf.sort_values(["team", "season", "week"])
    all_spread_perf["prev_avg_spread_perf"] = all_spread_perf.groupby(
        ["team", "season"]
    )["avg_spread_perf"].shift(1)

    # For week 1, use previous season's final average
    all_spread_perf["prev_avg_spread_perf"] = all_spread_perf.groupby("team")[
        "prev_avg_spread_perf"
    ].ffill()
    all_spread_perf["prev_avg_spread_perf"] = all_spread_perf[
        "prev_avg_spread_perf"
    ].fillna(0)

    # Create lookup table
    spread_perf_lookup = all_spread_perf[
        ["season", "week", "team", "prev_avg_spread_perf"]
    ].copy()

    # Step 4: Merge back to training data and calculate differential
    training_data = merge_most_recent_lookup(
        training_data,
        spread_perf_lookup,
        ["prev_avg_spread_perf"],
        "home_spread_perf_",
        "away_spread_perf_",
    )
    training_data.rename(
        columns={
            "home_spread_perf_prev_avg_spread_perf": "home_spread_perf",
            "away_spread_perf_prev_avg_spread_perf": "away_spread_perf",
        },
        inplace=True,
    )

    # Calculate the differential (home - away)
    training_data["spread_performance_diff"] = (
        training_data["home_spread_perf"] - training_data["away_spread_perf"]
    )

    # Drop intermediate columns
    training_data.drop(columns=["home_spread_perf", "away_spread_perf"], inplace=True)

    # Calculate last 3 games rolling average for spread performance
    print("Calculating spread performance differential (last 3 games)...")

    # Calculate rolling 3-game average for spread performance (crosses seasons)
    all_spread_perf_l3 = all_spread_perf.copy()
    all_spread_perf_l3 = all_spread_perf_l3.sort_values(["team", "season", "week"])

    # Use rolling window of 3, min_periods=1 to handle first few games
    all_spread_perf_l3["rolling_3game_spread"] = all_spread_perf_l3.groupby("team")[
        "spread_diff"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Shift to get previous week's 3-game average (don't include current week)
    all_spread_perf_l3["prev_rolling_3game_spread"] = all_spread_perf_l3.groupby(
        "team"
    )["rolling_3game_spread"].shift(1)

    # For first game of first season, fill with 0
    all_spread_perf_l3["prev_rolling_3game_spread"] = all_spread_perf_l3[
        "prev_rolling_3game_spread"
    ].fillna(0)

    # Create lookup table
    spread_perf_l3_lookup = all_spread_perf_l3[
        ["season", "week", "team", "prev_rolling_3game_spread"]
    ].copy()

    # Merge back to training data and calculate differential
    training_data = merge_most_recent_lookup(
        training_data,
        spread_perf_l3_lookup,
        ["prev_rolling_3game_spread"],
        "home_spread_perf_l3_",
        "away_spread_perf_l3_",
    )
    training_data.rename(
        columns={
            "home_spread_perf_l3_prev_rolling_3game_spread": "home_spread_perf_l3",
            "away_spread_perf_l3_prev_rolling_3game_spread": "away_spread_perf_l3",
        },
        inplace=True,
    )

    # Calculate the differential (home - away)
    training_data["spread_performance_diff_l3"] = (
        training_data["home_spread_perf_l3"] - training_data["away_spread_perf_l3"]
    )

    # Drop intermediate columns
    training_data.drop(
        columns=["home_spread_perf_l3", "away_spread_perf_l3"], inplace=True
    )

    # Calculate average margin of victory/defeat (actual point differential performance)
    print("Calculating average margin of victory differential...")

    # Calculate margin from each team's perspective (their score - opponent score)
    games_margin = games_df[
        ["season", "week", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    games_margin = games_margin.dropna(subset=["home_score", "away_score"])

    # Home team margin: home_score - away_score
    home_margin = games_margin[
        ["season", "week", "home_team", "home_score", "away_score"]
    ].copy()
    home_margin["actual_diff"] = home_margin["home_score"] - home_margin["away_score"]
    home_margin = home_margin[["season", "week", "home_team", "actual_diff"]]
    home_margin.columns = ["season", "week", "team", "actual_diff"]

    # Away team margin: away_score - home_score
    away_margin = games_margin[
        ["season", "week", "away_team", "away_score", "home_score"]
    ].copy()
    away_margin["actual_diff"] = away_margin["away_score"] - away_margin["home_score"]
    away_margin = away_margin[["season", "week", "away_team", "actual_diff"]]
    away_margin.columns = ["season", "week", "team", "actual_diff"]

    # Combine all margin records
    all_margins = pd.concat([home_margin, away_margin], ignore_index=True)
    all_margins = all_margins.sort_values(["team", "season", "week"])

    # Calculate cumulative average margin for each team
    all_margins["cumulative_margin"] = all_margins.groupby(["season", "team"])[
        "actual_diff"
    ].cumsum()
    all_margins["games_played"] = all_margins.groupby(["season", "team"]).cumcount() + 1
    all_margins["avg_margin"] = (
        all_margins["cumulative_margin"] / all_margins["games_played"]
    )

    # Shift to get previous week's average (don't include current week)
    all_margins = all_margins.sort_values(["team", "season", "week"])
    all_margins["prev_avg_margin"] = all_margins.groupby(["team", "season"])[
        "avg_margin"
    ].shift(1)

    # For week 1, use previous season's final average
    all_margins["prev_avg_margin"] = all_margins.groupby("team")[
        "prev_avg_margin"
    ].ffill()
    all_margins["prev_avg_margin"] = all_margins["prev_avg_margin"].fillna(0)

    # Create lookup table
    margin_lookup = all_margins[["season", "week", "team", "prev_avg_margin"]].copy()

    # Merge back to training data and calculate differential
    training_data = merge_most_recent_lookup(
        training_data,
        margin_lookup,
        ["prev_avg_margin"],
        "home_margin_",
        "away_margin_",
    )
    training_data.rename(
        columns={
            "home_margin_prev_avg_margin": "home_margin",
            "away_margin_prev_avg_margin": "away_margin",
        },
        inplace=True,
    )

    # Calculate the differential (home - away)
    training_data["avg_margin_of_victory_diff"] = (
        training_data["home_margin"] - training_data["away_margin"]
    )

    # Drop intermediate columns
    training_data.drop(columns=["home_margin", "away_margin"], inplace=True)

    # Calculate last 3 games rolling average for margin of victory
    print("Calculating average margin of victory differential (last 3 games)...")

    # Calculate rolling 3-game average for margin (crosses seasons)
    all_margins_l3 = all_margins.copy()
    all_margins_l3 = all_margins_l3.sort_values(["team", "season", "week"])

    # Use rolling window of 3, min_periods=1 to handle first few games
    all_margins_l3["rolling_3game_margin"] = all_margins_l3.groupby("team")[
        "actual_diff"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Shift to get previous week's 3-game average (don't include current week)
    all_margins_l3["prev_rolling_3game_margin"] = all_margins_l3.groupby("team")[
        "rolling_3game_margin"
    ].shift(1)

    # For first game of first season, fill with 0
    all_margins_l3["prev_rolling_3game_margin"] = all_margins_l3[
        "prev_rolling_3game_margin"
    ].fillna(0)

    # Create lookup table
    margin_l3_lookup = all_margins_l3[
        ["season", "week", "team", "prev_rolling_3game_margin"]
    ].copy()

    # Merge back to training data and calculate differential
    training_data = merge_most_recent_lookup(
        training_data,
        margin_l3_lookup,
        ["prev_rolling_3game_margin"],
        "home_margin_l3_",
        "away_margin_l3_",
    )
    training_data.rename(
        columns={
            "home_margin_l3_prev_rolling_3game_margin": "home_margin_l3",
            "away_margin_l3_prev_rolling_3game_margin": "away_margin_l3",
        },
        inplace=True,
    )

    # Calculate the differential (home - away)
    training_data["avg_margin_of_victory_diff_l3"] = (
        training_data["home_margin_l3"] - training_data["away_margin_l3"]
    )

    # Drop intermediate columns
    training_data.drop(columns=["home_margin_l3", "away_margin_l3"], inplace=True)

    # Reorder columns to match your specification
    final_columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "spread_line",
        "yahoo_spread",
        "rest_diff",
        "div_game",
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
        "completion_pct_diff",
        "yards_per_attempt_diff",
        "yards_per_carry_diff",
        "passing_epa_diff",
        "rushing_epa_diff",
        "power_ranking_diff",
        "win_pct_diff",
        "sos_diff",
        "adj_offensive_rank_diff",
        "adj_defensive_rank_diff",
        "adj_overall_rank_diff",
        "avg_weekly_point_diff",
        "spread_performance_diff",
        "avg_margin_of_victory_diff",
        "avg_weekly_point_diff_l3",
        "spread_performance_diff_l3",
        "avg_margin_of_victory_diff_l3",
        "power_ranking_diff_l3",
        "win_pct_diff_l3",
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

    # Only keep columns that exist
    final_columns = [col for col in final_columns if col in training_data.columns]
    training_data = training_data[final_columns]

    # Save to database
    engine = get_db_engine()

    # For each table, delete existing data for the backfill seasons, then append new data
    # This preserves historical data while updating only the specified seasons

    # Get the seasons being updated
    seasons_to_update = seasons  # This is the list of seasons passed to backfil_data
    seasons_str = ",".join(map(str, seasons_to_update))

    # Check if training_data table exists and has the correct schema
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='training_data'"
            )
        )
        table_exists = result.fetchone() is not None

    if table_exists:
        # Check if new columns exist
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(training_data)"))
            columns = [row[1] for row in result.fetchall()]

        if (
            "spread_performance_diff" not in columns
            or "avg_margin_of_victory_diff" not in columns
            or "spread_performance_diff_l3" not in columns
            or "avg_margin_of_victory_diff_l3" not in columns
            or "avg_weekly_point_diff_l3" not in columns
            or "power_ranking_diff_l3" not in columns
            or "win_pct_diff_l3" not in columns
            or "rest_diff" not in columns
            or "div_game" not in columns
            or "completion_pct_diff" not in columns
            or "yards_per_attempt_diff" not in columns
            or "yards_per_carry_diff" not in columns
            or "passing_epa_diff" not in columns
            or "rushing_epa_diff" not in columns
        ):
            print(
                "⚠️  Schema change detected - recreating training_data table with new columns..."
            )
            # Save existing data from other seasons before dropping
            other_seasons_data = pd.read_sql_query(
                text(
                    f"SELECT * FROM training_data WHERE season NOT IN ({seasons_str})"
                ),
                engine,
            )

            # Drop and recreate table
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS training_data"))
                conn.commit()

            # Insert new data for current seasons
            training_data.to_sql(
                "training_data", engine, if_exists="replace", index=False
            )

            # Append old data back if any exists
            if len(other_seasons_data) > 0:
                # Fill missing columns with 0 for old data
                if "spread_performance_diff" not in other_seasons_data.columns:
                    other_seasons_data["spread_performance_diff"] = 0.0
                if "avg_margin_of_victory_diff" not in other_seasons_data.columns:
                    other_seasons_data["avg_margin_of_victory_diff"] = 0.0
                if "spread_performance_diff_l3" not in other_seasons_data.columns:
                    other_seasons_data["spread_performance_diff_l3"] = 0.0
                if "avg_margin_of_victory_diff_l3" not in other_seasons_data.columns:
                    other_seasons_data["avg_margin_of_victory_diff_l3"] = 0.0
                if "avg_weekly_point_diff_l3" not in other_seasons_data.columns:
                    other_seasons_data["avg_weekly_point_diff_l3"] = 0.0
                if "power_ranking_diff_l3" not in other_seasons_data.columns:
                    other_seasons_data["power_ranking_diff_l3"] = 0.0
                if "win_pct_diff_l3" not in other_seasons_data.columns:
                    other_seasons_data["win_pct_diff_l3"] = 0.0
                other_seasons_data.to_sql(
                    "training_data", engine, if_exists="append", index=False
                )
                print(
                    f"   ✓ Restored {len(other_seasons_data)} rows from other seasons"
                )
        else:
            # Schema is correct, just delete and append normally
            with engine.connect() as conn:
                conn.execute(
                    text(f"DELETE FROM training_data WHERE season IN ({seasons_str})")
                )
                conn.commit()
            training_data.to_sql(
                "training_data", engine, if_exists="append", index=False
            )
    else:
        # Table doesn't exist, create it
        training_data.to_sql("training_data", engine, if_exists="replace", index=False)

    print(
        f"✓ Database updated for seasons: {min(seasons_to_update)}-{max(seasons_to_update)}"
    )
    print(f"  Historical data from other seasons preserved")


if __name__ == "__main__":
    print("backfilling data...")
    backfil_data(2003)
