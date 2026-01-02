import pickle

import numpy as np
import pandas as pd
import nflreadpy as nfl
from itertools import groupby

from src.model.features import engineer_features
from src.helpers.database_helpers import run_query, get_db_engine


def check_qb_change(season: int, week: int) -> str:

    query = f"""
        WITH TeamHistory AS (
            -- 1. Get Home Team History
            SELECT
                season,
                week,
                home_team AS team_name,
                home_qb_name AS qb_name
            FROM
                games
            WHERE
                season = {season} AND week < {week}

            UNION ALL

            -- 2. Get Away Team History
            SELECT
                season,
                week,
                away_team AS team_name,
                away_qb_name AS qb_name
            FROM
                games
            WHERE
                season = {season} AND week < {week}
        ),
        TeamQBLags AS (
            -- 3. Find the most recent QB for each team
            SELECT
                t1.team_name,
                -- The QB name from the row with the maximum week number
                t1.qb_name AS previous_qb_name
            FROM
                TeamHistory AS t1
            INNER JOIN (
                -- Find the maximum week (most recent game) for each team
                SELECT
                    team_name,
                    MAX(week) AS max_week
                FROM
                    TeamHistory
                GROUP BY 1
            ) AS t2 ON t1.team_name = t2.team_name AND t1.week = t2.max_week
        )
        -- 4. Join the lag data back to the original Week 15 games
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_qb_name,
            g.away_qb_name,
            g.home_team,
            g.away_team,
            -- Home QB Change Check
            CASE
                WHEN g.home_qb_name <> t_home.previous_qb_name THEN 'Y'
                ELSE 'N'
            END AS home_qb_changed,
            -- Away QB Change Check
            CASE
                WHEN g.away_qb_name <> t_away.previous_qb_name THEN 'Y'
                ELSE 'N'
            END AS away_qb_changed,
            t_home.previous_qb_name AS home_previous_qb,
            t_away.previous_qb_name AS away_previous_qb
        FROM
            games AS g
        -- Join for the HOME team's most recent previous QB
        LEFT JOIN
            TeamQBLags AS t_home
            ON g.home_team = t_home.team_name
        -- Join for the AWAY team's most recent previous QB
        LEFT JOIN
            TeamQBLags AS t_away
            ON g.away_team = t_away.team_name
        WHERE
            g.season = {season} AND g.week = {week}
        ORDER BY
            g.gameday,
            g.gametime;
    """
    results = run_query(query)
    results_df = pd.DataFrame(results)
    return results_df


def load_model(model_path: str = "nfl-prediction.pkl") -> dict:
    """
    Load the trained model and metadata.

    Returns:
        dict with keys: 'model', 'features', 'base_features', 'bias', 'metrics'
    """
    with open(model_path, "rb") as f:
        model_artifacts = pickle.load(f)

    return model_artifacts


def cover_spread_by(
    favorate: str, spread: float, model_winner: str, model_diff: float
) -> float:

    if model_diff == 0.8584951162338257:
        if favorate == model_winner:
            if spread < abs(model_diff):
                diff = abs(model_diff) - spread
            elif spread > abs(model_diff):
                diff = spread - abs(model_diff)
        else:
            if spread > abs(model_diff):
                diff = spread + abs(model_diff)
            elif spread < abs(model_diff):
                diff = spread + abs(model_diff)
        print(favorate, spread, model_winner, model_diff, diff)

    if favorate == model_winner:
        if spread < abs(model_diff):
            return abs(model_diff) - spread
        elif spread > abs(model_diff):
            return spread - abs(model_diff)
    else:
        if spread > abs(model_diff):
            return spread + abs(model_diff)
        elif spread < abs(model_diff):
            return spread + abs(model_diff)
    return 0


def get_future_predictions(season=2025):

    conn = get_db_engine()

    model_artifacts = load_model("nfl-prediction.pkl")
    week = nfl.get_current_week()

    qb_change = check_qb_change(season, week)

    model = model_artifacts["model"]
    features = model_artifacts["features"]

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == {season} and td.week == {week} AND (away_score IS NULL OR home_score IS NULL)"
    )
    game_df = pd.DataFrame(game_to_predict)

    predictions = []

    for game in game_df.iterrows():
        spread = game[1]["yahoo_spread"]

        if spread < 0:
            spread = abs(spread)
            fav = "home_team"
        else:
            spread = abs(spread)
            fav = "away_team"

        predict_df = pd.DataFrame([game[1]])
        predict_df.drop(
            columns=["game_id", "away_score", "home_score", "season"], inplace=True
        )
        df, _ = engineer_features(predict_df)
        X = df[features]
        prediction = model.predict(X)

        pidicted_diff = prediction[0]

        if fav == "home_team":
            favorate_team = game[1]["home_team"]
            if pidicted_diff > 0:
                predicted_winner = game[1]["home_team"]
                pred_home_score = prediction[0]
                pred_away_score = spread
            else:
                predicted_winner = game[1]["away_team"]
                pred_home_score = 0
                pred_away_score = abs(prediction[0]) + spread
        else:
            favorate_team = game[1]["away_team"]
            if pidicted_diff < 0:
                predicted_winner = game[1]["away_team"]
                pred_away_score = abs(prediction[0])
                pred_home_score = spread
            else:
                predicted_winner = game[1]["home_team"]
                pred_away_score = 0
                pred_home_score = prediction[0] + spread

        if pred_home_score > pred_away_score:
            predicted_winner_spread = game[1]["home_team"]
        else:
            predicted_winner_spread = game[1]["away_team"]

        # Calculate turnover differential from individual columns
        turnovers_forced = (
            game[1].get("def_passing_interceptions_diff", 0)
            + game[1].get("def_sack_fumbles_lost_diff", 0)
            + game[1].get("def_rushing_fumbles_lost_diff", 0)
        )
        turnovers_committed = (
            game[1].get("passing_interceptions_diff", 0)
            + game[1].get("sack_fumbles_lost_diff", 0)
            + game[1].get("rushing_fumbles_lost_diff", 0)
        )
        turnover_diff = turnovers_forced - turnovers_committed

        predictions.append(
            {
                "game_id": game[1]["game_id"],
                "season": season,
                "week": week,
                "home_team": game[1]["home_team"],
                "away_team": game[1]["away_team"],
                "spread": spread,
                "spread_favorite": favorate_team,
                "div_game": game[1]["div_game"],
                "predicted_winner": predicted_winner_spread,
                "margin": float(
                    cover_spread_by(
                        game[1][fav], spread, predicted_winner_spread, pidicted_diff
                    ),
                ),
                "power_ranking_diff": game[1]["power_ranking_diff"],
                "win_pct_diff": game[1]["win_pct_diff"],
                "avg_weekly_point_diff_l3": game[1]["avg_weekly_point_diff_l3"],
                "passing_epa_diff": game[1]["passing_epa_diff"],
                "rushing_epa_diff": game[1]["rushing_epa_diff"],
                "adj_overall_rank_diff": game[1]["adj_overall_rank_diff"],
                "home_qb_changed": qb_change.loc[
                    (qb_change["game_id"] == game[1]["game_id"]), "home_qb_changed"
                ].values[0],
                "away_qb_changed": qb_change.loc[
                    (qb_change["game_id"] == game[1]["game_id"]), "away_qb_changed"
                ].values[0],
                "power_ranking_diff_l3": game[1]["power_ranking_diff_l3"],
                "avg_margin_of_victory_diff": game[1]["avg_margin_of_victory_diff"],
                "avg_margin_of_victory_diff_l3": game[1][
                    "avg_margin_of_victory_diff_l3"
                ],
                "avg_weekly_point_diff": game[1]["avg_weekly_point_diff"],
                "spread_performance_diff": game[1]["spread_performance_diff"],
                "sacks_suffered_avg_diff": game[1]["sacks_suffered_avg_diff"],
                "completion_pct_diff": game[1].get("completion_pct_diff", None),
                "yards_per_attempt_diff": game[1].get("yards_per_attempt_diff", None),
                "yards_per_carry_diff": game[1].get("yards_per_carry_diff", None),
                "turnover_diff": turnover_diff,
                "rest_diff": game[1].get("rest_diff", None),
                "spread_diff": abs(
                    (game[1]["spread_line"] * -1) - game[1]["yahoo_spread"]
                ),
                "rushing_yards_diff": game[1].get("rushing_yards_diff", None),
                "def_rushing_yards_diff": game[1].get("def_rushing_yards_diff", None),
                "passing_yards_diff": game[1].get("passing_yards_diff", None),
            }
        )

    sorted_predictions = sorted(predictions, key=lambda x: x["margin"], reverse=True)
    for i, item in enumerate(sorted_predictions):
        item["confidence"] = i + 1
    for p in sorted_predictions:
        warning = ""
        if (p["home_qb_changed"] == "Y") or (p["away_qb_changed"] == "Y"):
            if p["home_qb_changed"] == "Y":
                warning += f" - Warning: {p['home_team']} QB CHANGE"
            elif p["away_qb_changed"] == "Y":
                warning += f" - Warning: {p['away_team']} QB CHANGE"
        print(
            f"Predicted winner over the {p['spread']} spread: \t{p['predicted_winner']} \twith confidence rank {p['confidence']} and margin {p['margin']:.2f}   \t{warning}"
        )
    df = pd.DataFrame(predictions)
    df["margin"] = df["margin"].astype(float)
    df.to_sql("future_predictions", conn, if_exists="replace", index=False)


def get_past_predictions(season=2025):

    conn = get_db_engine()

    model_artifacts = load_model("nfl-prediction.pkl")
    current_week = nfl.get_current_week()

    model = model_artifacts["model"]
    features = model_artifacts["features"]

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == {season} and td.week <= {current_week} AND (away_score IS NOT NULL OR home_score IS NOT NULL) order by week asc"
    )
    game_df = pd.DataFrame(game_to_predict)
    predictions = []

    correct = 0
    total = 0
    week = 1

    overall_correct = 0
    overall_correct_total = 0

    for game in game_df.iterrows():
        qb_change = check_qb_change(season, game[1]["week"])
        if game[1]["week"] != week:
            print(
                f"Week {week}: {correct}/{total} ({(correct/total*100) if total > 0 else 0:.2f}%)"
            )
            week = game[1]["week"]
            correct = 0
            total = 0

        spread = game[1]["yahoo_spread"]

        if spread < 0:
            spread = abs(spread)
            diff = (game[1]["home_score"] - spread) - game[1]["away_score"]
            fav = "home_team"
        else:
            spread = abs(spread)
            diff = game[1]["home_score"] - (game[1]["away_score"] - spread)
            fav = "away_team"

        if diff > 0:
            winner_over_spread = game[1]["home_team"]
        elif diff == 0:
            continue
        else:
            winner_over_spread = game[1]["away_team"]

        predict_df = pd.DataFrame([game[1]])
        predict_df.drop(
            columns=["game_id", "away_score", "home_score", "season", "yahoo_spread"],
            inplace=True,
        )
        df, _ = engineer_features(predict_df)
        X = df[features]
        prediction = model.predict(X)

        pidicted_diff = prediction[0]

        if fav == "home_team":
            favorate_team = game[1]["home_team"]
            if pidicted_diff > 0:
                pred_home_score = prediction[0]
                pred_away_score = spread
            else:
                pred_home_score = 0
                pred_away_score = abs(prediction[0]) + spread
        else:
            favorate_team = game[1]["away_team"]
            if pidicted_diff < 0:
                pred_away_score = abs(prediction[0])
                pred_home_score = spread
            else:
                pred_away_score = 0
                pred_home_score = prediction[0] + spread

        if pred_home_score > pred_away_score:
            predicted_winner = game[1]["home_team"]
        else:
            predicted_winner = game[1]["away_team"]

        if predicted_winner == winner_over_spread:
            correct += 1
            overall_correct += 1
            total += 1
            overall_correct_total += 1
        else:
            total += 1
            overall_correct_total += 1

        predictions.append(
            {
                "game_id": game[1]["game_id"],
                "season": season,
                "week": week,
                "home_team": game[1]["home_team"],
                "away_team": game[1]["away_team"],
                "spread": spread,
                "spread_favorite": favorate_team,
                "predicted_winner": predicted_winner,
                "margin": float(
                    cover_spread_by(
                        game[1][fav], spread, predicted_winner, pidicted_diff
                    )
                ),
                "correct": predicted_winner == winner_over_spread,
                "power_ranking_diff": game[1]["power_ranking_diff"],
                "win_pct_diff": game[1]["win_pct_diff"],
                "avg_weekly_point_diff_l3": game[1]["avg_weekly_point_diff_l3"],
                "passing_epa_diff": game[1]["passing_epa_diff"],
                "rushing_epa_diff": game[1]["rushing_epa_diff"],
                "adj_overall_rank_diff": game[1]["adj_overall_rank_diff"],
                "home_qb_changed": qb_change.loc[
                    (qb_change["game_id"] == game[1]["game_id"]), "home_qb_changed"
                ].values[0],
                "away_qb_changed": qb_change.loc[
                    (qb_change["game_id"] == game[1]["game_id"]), "away_qb_changed"
                ].values[0],
                "power_ranking_diff_l3": game[1]["power_ranking_diff_l3"],
                "avg_margin_of_victory_diff": game[1]["avg_margin_of_victory_diff"],
                "avg_margin_of_victory_diff_l3": game[1][
                    "avg_margin_of_victory_diff_l3"
                ],
                "avg_weekly_point_diff": game[1]["avg_weekly_point_diff"],
                "spread_performance_diff": game[1]["spread_performance_diff"],
                "sacks_suffered_avg_diff": game[1]["sacks_suffered_avg_diff"],
                "completion_pct_diff": game[1].get("completion_pct_diff", None),
                "yards_per_attempt_diff": game[1].get("yards_per_attempt_diff", None),
                "yards_per_carry_diff": game[1].get("yards_per_carry_diff", None),
                "turnover_diff": (
                    game[1].get("def_passing_interceptions_diff", 0)
                    + game[1].get("def_sack_fumbles_lost_diff", 0)
                    + game[1].get("def_rushing_fumbles_lost_diff", 0)
                    - game[1].get("passing_interceptions_diff", 0)
                    - game[1].get("sack_fumbles_lost_diff", 0)
                    - game[1].get("rushing_fumbles_lost_diff", 0)
                ),
                "rest_diff": game[1].get("rest_diff", None),
                "spread_diff": abs(
                    (game[1]["spread_line"] * -1) - game[1]["yahoo_spread"]
                ),
                "rushing_yards_diff": game[1].get("rushing_yards_diff", None),
                "def_rushing_yards_diff": game[1].get("def_rushing_yards_diff", None),
                "passing_yards_diff": game[1].get("passing_yards_diff", None),
            }
        )

    print(
        f"Week {week}: {correct}/{total} ({(correct/total*100) if total > 0 else 0:.2f}%)"
    )
    print(
        f"Overall Spread Prediction Accuracy for 2025 so far: {overall_correct}/{overall_correct_total} ({((overall_correct)/(overall_correct_total)*100) if (overall_correct_total) > 0 else 0:.2f}%)"
    )

    sorted_predictions = sorted(
        predictions, key=lambda x: (x["week"], x["margin"]), reverse=True
    )
    for week, games in groupby(sorted_predictions, key=lambda x: x["week"]):
        games_list = list(games)
        for i, game in enumerate(games_list):
            game["confidence"] = i + 1

    df = pd.DataFrame(predictions)
    df["margin"] = df["margin"].astype(float)
    df.to_sql("past_predictions", conn, if_exists="replace", index=False)

    return (
        (overall_correct) / (overall_correct_total) * 100
        if (overall_correct_total) > 0
        else 0
    )


def get_past_predictions_model(model):
    """Get week-by-week 2025 results with consistency metrics."""
    features = model["features"]
    current_week = nfl.get_current_week()

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == 2025 and td.week <= {current_week} AND (away_score IS NOT NULL OR home_score IS NOT NULL) order by week asc"
    )
    game_df = pd.DataFrame(game_to_predict)

    week_results = {}
    overall_correct = 0
    overall_total = 0

    for _, game in game_df.iterrows():
        week = game["week"]
        spread = game["yahoo_spread"]

        if spread < 0:
            spread = abs(spread)
            diff = (game["home_score"] - spread) - game["away_score"]
            fav = "home_team"
        else:
            spread = abs(spread)
            diff = game["home_score"] - (game["away_score"] - spread)
            fav = "away_team"

        if diff > 0:
            winner_over_spread = game["home_team"]
        elif diff == 0:
            continue
        else:
            winner_over_spread = game["away_team"]

        predict_df = pd.DataFrame([game])
        predict_df.drop(
            columns=["game_id", "away_score", "home_score", "season", "yahoo_spread"],
            inplace=True,
        )
        df, _ = engineer_features(predict_df)
        X = df[features]
        prediction = model["model"].predict(X)

        pidicted_diff = prediction[0]

        if fav == "home_team":
            if pidicted_diff > 0:
                pred_home_score = prediction[0]
                pred_away_score = spread
            else:
                pred_home_score = 0
                pred_away_score = abs(prediction[0]) + spread
        else:
            if pidicted_diff < 0:
                pred_away_score = abs(prediction[0])
                pred_home_score = spread
            else:
                pred_away_score = 0
                pred_home_score = prediction[0] + spread

        if pred_home_score > pred_away_score:
            predicted_winner = game["home_team"]
        else:
            predicted_winner = game["away_team"]

        if week not in week_results:
            week_results[week] = {"correct": 0, "total": 0}

        if predicted_winner == winner_over_spread:
            week_results[week]["correct"] += 1
            overall_correct += 1

        week_results[week]["total"] += 1
        overall_total += 1

    weekly_accuracies = []
    for week, results in week_results.items():
        if results["total"] > 0:
            accuracy = (results["correct"] / results["total"]) * 100
            weekly_accuracies.append(accuracy)

    overall_accuracy = (
        (overall_correct / overall_total * 100) if overall_total > 0 else 0
    )

    consistency_std = np.std(weekly_accuracies) if len(weekly_accuracies) > 1 else 0
    weeks_above_50 = sum(1 for acc in weekly_accuracies if acc >= 50)
    min_week_accuracy = min(weekly_accuracies) if weekly_accuracies else 0

    return {
        "overall_accuracy": overall_accuracy,
        "weekly_accuracies": weekly_accuracies,
        "consistency_std": consistency_std,
        "weeks_above_50": weeks_above_50,
        "total_weeks": len(weekly_accuracies),
        "min_week_accuracy": min_week_accuracy,
        "week_results": week_results,
    }


# def predict(game_df, model):

#     week = game_df["week"]
#     spread = game_df["yahoo_spread"]

#     if spread < 0:
#         spread = abs(spread)
#         diff = (game_df["home_score"] - spread) - game_df["away_score"]
#         fav = "home_team"
#     else:
#         spread = abs(spread)
#         diff = game_df["home_score"] - (game_df["away_score"] - spread)
#         fav = "away_team"

#     if diff > 0:
#         winner_over_spread = game_df["home_team"]
#     elif diff == 0:
#         winner_over_spread = "PUSH"
#     else:
#         winner_over_spread = game_df["away_team"]

#     predict_df = pd.DataFrame([game_df])
#     predict_df.drop(
#         columns=["game_id", "away_score", "home_score", "season", "yahoo_spread"],
#         inplace=True,
#     )
#     df, _ = engineer_features(predict_df)
#     X = df[features]
#     prediction = model["model"].predict(X)

#     pidicted_diff = prediction[0]

#     if fav == "home_team":
#         if pidicted_diff > 0:
#             pred_home_score = prediction[0]
#             pred_away_score = spread
#         else:
#             pred_home_score = 0
#             pred_away_score = abs(prediction[0]) + spread
#     else:
#         if pidicted_diff < 0:
#             pred_away_score = abs(prediction[0])
#             pred_home_score = spread
#         else:
#             pred_away_score = 0
#             pred_home_score = prediction[0] + spread
