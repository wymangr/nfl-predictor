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
    favorite: str, spread: float, model_winner: str, model_diff: float
) -> float:

    if favorite == model_winner:
        if spread < abs(model_diff):
            return abs(model_diff) - spread
        elif spread > abs(model_diff):
            return spread + abs(model_diff)
    else:
        if spread > abs(model_diff):
            return spread - abs(model_diff)
        elif spread < abs(model_diff):
            return spread + abs(model_diff)
    return 0


def get_future_predictions(season=2025):

    conn = get_db_engine()
    model_artifacts = load_model("nfl-prediction.pkl")
    week = nfl.get_current_week()

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == {season} and td.week == {week} AND (away_score IS NULL OR home_score IS NULL) AND td.yahoo_spread IS NOT NULL"
    )
    games_df = pd.DataFrame(game_to_predict)
    predictions = predict(games_df, model_artifacts)

    sorted_predictions = sorted(
        predictions, key=lambda x: x["cover_spread_by"], reverse=True
    )
    print("If Predicted diff is positive, home team is expected to win.")
    for i, item in enumerate(sorted_predictions):
        item["confidence"] = i + 1
    for p in sorted_predictions:
        warning = ""
        if (p["home_qb_changed"] == "Y") or (p["away_qb_changed"] == "Y"):
            if p["home_qb_changed"] == "Y":
                warning += f" - Warning: {p['home_team']} QB CHANGE"
            elif p["away_qb_changed"] == "Y":
                warning += f" - Warning: {p['away_team']} QB CHANGE"
        if p["predicted_winner"] == p["home_team"]:
            home_away = "Home"
        else:
            home_away = "Away"
        print(
            f"Predicted winner over the {p['spread']} spread: \t{p['predicted_winner']} ({home_away}) \twith confidence rank {p['confidence']} by {p['cover_spread_by']:.2f} - Predicted diff: {p['predicted_diff']:.2f}   \t{warning}"
        )
    df = pd.DataFrame(predictions)
    df.to_sql("future_predictions", conn, if_exists="replace", index=False)


def get_past_predictions(season=2025):

    conn = get_db_engine()

    model_artifacts = load_model("nfl-prediction.pkl")
    if season != nfl.get_current_season():
        current_week = run_query(f"SELECT MAX(week) FROM training_data WHERE season = {season}")[0]['MAX(week)']
        if not current_week:
            print(f"No data for season {season}.")
            return 0
    else:
        current_week = nfl.get_current_week()

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == {season} and td.week <= {current_week} AND (away_score IS NOT NULL OR home_score IS NOT NULL) AND td.yahoo_spread IS NOT NULL order by week asc"
    )
    games_df = pd.DataFrame(game_to_predict)
    if games_df.empty:
        print("No games to predict.")
        return 0

    correct = 0
    total = 0
    week = 1

    overall_correct = 0
    overall_correct_total = 0

    predictions = predict(games_df, model_artifacts)

    for prediction in sorted(predictions, key=lambda x: (x["week"])):
        if prediction["week"] != week:
            print(
                f"Week {week}: {correct}/{total} ({(correct/total*100) if total > 0 else 0:.2f}%)"
            )
            week = prediction["week"]
            correct = 0
            total = 0
        if prediction["correct"]:
            if prediction["correct"] == "push":
                pass
            else:
                correct += 1
                overall_correct += 1
                overall_correct_total += 1
                total += 1
        else:
            total += 1
            overall_correct_total += 1

    print(
        f"Week {week}: {correct}/{total} ({(correct/total*100) if total > 0 else 0:.2f}%)"
    )
    print(
        f"Overall Spread Prediction Accuracy for 2025 so far: {overall_correct}/{overall_correct_total} ({((overall_correct)/(overall_correct_total)*100) if (overall_correct_total) > 0 else 0:.2f}%)"
    )

    sorted_predictions = sorted(
        predictions, key=lambda x: (x["week"], x["cover_spread_by"]), reverse=True
    )
    for week, games in groupby(sorted_predictions, key=lambda x: x["week"]):
        games_list = list(games)
        for i, game in enumerate(games_list):
            game["confidence"] = i + 1

    df = pd.DataFrame(predictions)
    df.to_sql("past_predictions", conn, if_exists="replace", index=False)

    return (
        (overall_correct) / (overall_correct_total) * 100
        if (overall_correct_total) > 0
        else 0
    )


def get_past_predictions_model(model):
    """Get week-by-week 2025 results with consistency metrics."""
    current_week = nfl.get_current_week()

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == 2025 and td.week <= {current_week} AND (away_score IS NOT NULL OR home_score IS NOT NULL) order by week asc"
    )
    games_df = pd.DataFrame(game_to_predict)

    correct = 0
    total = 0
    week = 1

    overall_correct = 0
    overall_correct_total = 0

    predictions = predict(games_df, model)

    week_results = {}
    weekly_accuracies = []
    for prediction in sorted(predictions, key=lambda x: (x["week"])):

        if prediction["week"] != week:
            weekly_accuracies.append(correct / total * 100)
            week = prediction["week"]
            correct = 0
            total = 0

        if week not in week_results:
            week_results[week] = {"correct": 0, "total": 0}

        if prediction["correct"]:
            if prediction["correct"] == "push":
                pass
            else:
                correct += 1
                overall_correct += 1
                overall_correct_total += 1
                total += 1
                week_results[week]["correct"] += 1
        else:
            total += 1
            overall_correct_total += 1

    overall_accuracy = (
        (overall_correct / overall_correct_total * 100)
        if overall_correct_total > 0
        else 0
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


def check_correct_prediction(
    spread, home_score, away_score, home_team, away_team, predicted_winner
):
    if spread < 0:
        diff = (home_score - abs(spread)) - away_score
    else:
        diff = home_score - (away_score - abs(spread))

    if diff > 0:
        winner_over_spread = home_team
    elif diff == 0:
        return "push"
    else:
        winner_over_spread = away_team

    return predicted_winner == winner_over_spread


def predict(games_df, model):

    predictions = []

    for game in games_df.iterrows():
        features = model["features"]
        season = game[1]["season"]
        week = game[1]["week"]
        spread = game[1]["yahoo_spread"]
        qb_change = check_qb_change(season, week)

        if spread < 0:
            spread_favorite = game[1]["home_team"]
            abs_spread = abs(spread)
            fav = "home_team"
        else:
            spread_favorite = game[1]["away_team"]
            abs_spread = abs(spread)
            fav = "away_team"

        game_id = game[1]["game_id"]
        home_score = game[1]["home_score"]
        away_score = game[1]["away_score"]

        predict_df = pd.DataFrame([game[1]])
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
                pred_away_score = abs_spread
            else:
                pred_home_score = 0
                pred_away_score = abs(prediction[0]) + abs_spread
        else:
            if pidicted_diff < 0:
                pred_away_score = abs(prediction[0])
                pred_home_score = abs_spread
            else:
                pred_away_score = 0
                pred_home_score = prediction[0] + abs_spread

        if pred_home_score > pred_away_score:
            predicted_winner = game[1]["home_team"]
        else:
            predicted_winner = game[1]["away_team"]

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

        correct = None
        if home_score is not None and away_score is not None:
            correct = check_correct_prediction(
                spread,
                home_score,
                away_score,
                game[1]["home_team"],
                game[1]["away_team"],
                predicted_winner,
            )

        prediction = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "home_team": game[1]["home_team"],
            "away_team": game[1]["away_team"],
            "spread": abs_spread,
            "spread_favorite": spread_favorite,
            "div_game": game[1]["div_game"],
            "predicted_winner": predicted_winner,
            "predicted_diff": float(pidicted_diff),
            "cover_spread_by": float(
                cover_spread_by(
                    game[1][fav], abs_spread, predicted_winner, pidicted_diff
                ),
            ),
            "power_ranking_diff": game[1]["power_ranking_diff"],
            "win_pct_diff": game[1]["win_pct_diff"],
            "avg_weekly_point_diff_l3": game[1]["avg_weekly_point_diff_l3"],
            "passing_epa_diff": game[1]["passing_epa_diff"],
            "rushing_epa_diff": game[1]["rushing_epa_diff"],
            "adj_overall_rank_diff": game[1]["adj_overall_rank_diff"],
            "home_qb_changed": qb_change.loc[
                (qb_change["game_id"] == game_id), "home_qb_changed"
            ].values[0],
            "away_qb_changed": qb_change.loc[
                (qb_change["game_id"] == game_id), "away_qb_changed"
            ].values[0],
            "power_ranking_diff_l3": game[1]["power_ranking_diff_l3"],
            "avg_margin_of_victory_diff": game[1]["avg_margin_of_victory_diff"],
            "avg_margin_of_victory_diff_l3": game[1]["avg_margin_of_victory_diff_l3"],
            "avg_weekly_point_diff": game[1]["avg_weekly_point_diff"],
            "spread_performance_diff": game[1]["spread_performance_diff"],
            "sacks_suffered_avg_diff": game[1]["sacks_suffered_avg_diff"],
            "completion_pct_diff": game[1].get("completion_pct_diff", None),
            "yards_per_attempt_diff": game[1].get("yards_per_attempt_diff", None),
            "yards_per_carry_diff": game[1].get("yards_per_carry_diff", None),
            "turnover_diff": turnover_diff,
            "rest_diff": game[1].get("rest_diff", None),
            "spread_diff": abs((game[1]["spread_line"] * -1) - spread),
            "rushing_yards_diff": game[1].get("rushing_yards_diff", None),
            "def_rushing_yards_diff": game[1].get("def_rushing_yards_diff", None),
            "passing_yards_diff": game[1].get("passing_yards_diff", None),
            "correct": correct,
        }
        predictions.append(prediction)
    return predictions
