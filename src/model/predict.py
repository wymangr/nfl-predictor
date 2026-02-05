import pickle
import math

import numpy as np
import pandas as pd
import nflreadpy as nfl
from itertools import groupby

from src.model.features import engineer_features
from src.helpers.database_helpers import run_query, get_db_engine
from src.reports.past_predictions_analysis import (
    get_bucket_analysis,
    classify_prediction_bucket,
    calculate_bucket_confidence,
)


def get_confidence_score(game_data, uncertainty=None) -> float:
    """
    Calculate confidence score for a game prediction.

    This is the single source of truth for the confidence score formula.
    When optimizing, update this function with the new formula.

    Two modes of operation:
    1. If uncertainty is provided: directly converts uncertainty to confidence score
    2. If game_data is provided: calculates uncertainty from features, then converts to confidence

    Args:
        game_data: Dictionary/Series with game features including:
                   - spread (absolute value)
                   - cover_spread_by (can be positive or negative)
                   - other game features
                   Can be None if uncertainty is provided directly.
        uncertainty: Optional pre-calculated uncertainty value.
                     If provided, skips feature-based calculation.

    Returns:
        float: confidence score value (0-100)
    """
    FIXED_MAX_UNCERTAINTY = 10.0
    if uncertainty is None:
        uncertainty = (
            game_data["spread"] / 11 * 0.243
            + abs(game_data["cover_spread_by"]) * 0.186
            + abs(game_data["yards_per_carry_diff"]) * 0.067
            + abs(game_data["predicted_diff"]) / 5 * 0.068
            + abs(game_data["rushing_epa_diff"]) / 50 * 0.226
            + abs(game_data["power_ranking_diff_l3"]) * 0.210
        )

    confidence_score = 100 - min((uncertainty / FIXED_MAX_UNCERTAINTY * 100), 100)
    return confidence_score


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


def confidence_accuracy(confidence_score: float) -> str:
    confidence_accuracy = run_query(
        f"""SELECT 
            AVG(CAST(correct AS FLOAT)) * 100 AS accuracy_percentage,
            COUNT(*) AS sample_size
        FROM past_predictions
        WHERE confidence_score BETWEEN ({confidence_score} - 1) AND ({confidence_score} + 1)
            AND correct != 'push';"""
    )
    if confidence_accuracy[0]["sample_size"] == 0:
        return math.nan, 0

    return (
        confidence_accuracy[0]["accuracy_percentage"],
        confidence_accuracy[0]["sample_size"],
    )


def get_bucket_confidence_accuracy(prediction_dict, bucket_df):
    """
    Calculate bucket-based confidence and accuracy for a single prediction.

    Args:
        prediction_dict: Dictionary with prediction metrics
        bucket_df: DataFrame with bucket analysis data

    Returns:
        tuple: (bucket_confidence, bucket_accuracy, sample_size) or (None, None, 0) if unable to calculate
    """
    # Use shared function to calculate bucket confidence
    weighted_acc, bucket_details = calculate_bucket_confidence(
        prediction_dict, bucket_df
    )

    if weighted_acc is None:
        return None, None, 0

    # Now calculate all past predictions' bucket confidence to determine 1% bracket accuracy
    # Get all past predictions with their metrics
    past_games = run_query(
        """
        SELECT *
        FROM past_predictions
        WHERE correct != 'push'
    """
    )

    if not past_games:
        return weighted_acc, None, 0

    # Calculate bucket confidence for each past game using shared function
    past_bucket_confidences = []
    for game in past_games:
        # Use shared function to calculate bucket confidence for this past game
        game_weighted_acc, _ = calculate_bucket_confidence(game, bucket_df)

        if game_weighted_acc is not None:
            past_bucket_confidences.append(
                {
                    "bucket_confidence": game_weighted_acc,
                    "correct": game["correct"] == "1",
                }
            )

    # Find which 1% bracket this game falls into
    bucket_min = int(weighted_acc)
    bucket_max = bucket_min + 1

    # Calculate accuracy for this bracket
    bracket_games = [
        p
        for p in past_bucket_confidences
        if p["bucket_confidence"] >= bucket_min and p["bucket_confidence"] < bucket_max
    ]

    if bracket_games:
        bracket_correct = sum(1 for p in bracket_games if p["correct"])
        bracket_total = len(bracket_games)
        bracket_accuracy = (bracket_correct / bracket_total) * 100

        return weighted_acc, bracket_accuracy, bracket_total

    return weighted_acc, None, 0


def get_future_predictions(spread_line=False, bucket=False):

    conn = get_db_engine()
    model_artifacts = load_model("nfl-prediction.pkl")
    week = nfl.get_current_week()
    season = nfl.get_current_season()

    # Run past predictions to update past data for confidence metrics
    get_past_predictions(season="all", spread_line=spread_line, quiet=True)

    # Load bucket analysis if requested
    bucket_df = None
    if bucket:
        print("Loading bucket analysis for confidence calculation...")
        bucket_df = get_bucket_analysis("past_predictions")

    if spread_line:
        print("Using nflreadpy spread line for predictions.")

    games_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == {season} and td.week == {week} AND (away_score IS NULL OR home_score IS NULL) AND td.yahoo_spread IS NOT NULL"
    )
    if len(games_to_predict) == 0:
        print(f"No future games to predict for Week {week}, {season}.")
        return

    games_df = pd.DataFrame(games_to_predict)
    predictions = predict(games_df, model_artifacts, spread_line)

    sorted_predictions = sorted(
        predictions, key=lambda x: x["confidence"], reverse=False
    )
    print("If Predicted diff is positive, home team is expected to win.")
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

        confidence_acc, sample_size = confidence_accuracy(p["confidence_score"])

        output = (
            f"Pick: {p['predicted_winner']} ({home_away}) \t"
            f"Spread: {p['spread']} (Fav {p['spread_favorite']})\t"
            f"Cover by: {p['cover_spread_by']:.2f}\t"
            f"Predicted Diff: {p['predicted_diff']:.2f}\t"
            f"Confidence: {p['confidence']}\t"
            f"Confidence Score: {p['confidence_score']:.2f} \t"
            f"Confidence Accuracy: {confidence_acc:.2f}% (Sample Size: {sample_size})\t"
        )

        # Add bucket accuracy if requested
        if bucket and bucket_df is not None:
            bucket_conf, bucket_acc, bucket_sample = get_bucket_confidence_accuracy(
                p, bucket_df
            )
            if bucket_acc is not None:
                output += f"Bucket Accuracy: {bucket_acc:.1f}% (Bucket: {bucket_conf:.1f}%, n={bucket_sample})\t"

        output += warning
        print(output)

    df = pd.DataFrame(predictions)
    df.to_sql("future_predictions", conn, if_exists="replace", index=False)


def get_past_predictions(season="2025", spread_line=False, quiet=False):
    """
    Generate predictions for past games.

    Args:
        season: Can be a single year (2025 or "2025"), multiple years ("2024,2025"), or "all"
        spread_line: Whether to use spread_line or yahoo_spread
        quiet: Whether to suppress output

    Returns:
        Overall accuracy percentage
    """
    # Parse season parameter
    if isinstance(season, int):
        seasons = [season]
    elif str(season).lower() == "all":
        # Get all available seasons from database
        available_seasons = run_query(
            "SELECT DISTINCT season FROM training_data ORDER BY season"
        )
        seasons = [s["season"] for s in available_seasons]
        if not quiet:
            print(f"Processing all available seasons: {', '.join(map(str, seasons))}")
    else:
        # Parse comma-separated years
        seasons = [int(y.strip()) for y in str(season).split(",")]

    spread_column = "yahoo_spread"
    conn = get_db_engine()
    if spread_line:
        if not quiet:
            print("Using nflreadpy spread line for predictions.")
        spread_column = "spread_line"

    model_artifacts = load_model("nfl-prediction.pkl")

    all_predictions = []
    season_accuracies = {}

    for season_year in seasons:
        if not quiet and len(seasons) > 1:
            print(f"\n{'='*70}")
            print(f"Processing Season {season_year}")
            print(f"{'='*70}")

        if season_year != nfl.get_current_season():
            current_week = run_query(
                f"SELECT MAX(week) FROM training_data WHERE season = {season_year}"
            )[0]["MAX(week)"]
            if not current_week:
                if not quiet:
                    print(f"No data for season {season_year} to make past predictions.")
                continue
        else:
            current_week = nfl.get_current_week()

        game_to_predict = run_query(
            f"SELECT * FROM training_data td where td.season == {season_year} and td.week <= {current_week} AND (away_score IS NOT NULL OR home_score IS NOT NULL) AND td.{spread_column} IS NOT NULL order by week asc"
        )
        games_df = pd.DataFrame(game_to_predict)
        if games_df.empty:
            if not quiet:
                print(f"No past games to predict for season {season_year}.")
            continue

        correct = 0
        total = 0
        week = 1

        overall_correct = 0
        overall_correct_total = 0

        predictions = predict(games_df, model_artifacts, spread_line)

        for prediction in sorted(predictions, key=lambda x: (x["week"])):
            if prediction["week"] != week:
                if not quiet:
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

        if not quiet:
            print(
                f"Week {week}: {correct}/{total} ({(correct/total*100) if total > 0 else 0:.2f}%)"
            )
            print(
                f"Overall Spread Prediction Accuracy for {season_year}: {overall_correct}/{overall_correct_total} ({((overall_correct)/(overall_correct_total)*100) if (overall_correct_total) > 0 else 0:.2f}%)"
            )

        season_accuracy = (
            (overall_correct) / (overall_correct_total) * 100
            if (overall_correct_total) > 0
            else 0
        )
        season_accuracies[season_year] = season_accuracy
        all_predictions.extend(predictions)

    # Save all predictions to database
    if all_predictions:
        df = pd.DataFrame(all_predictions)
        df.to_sql("past_predictions", conn, if_exists="replace", index=False)

        # Print summary if multiple seasons
        if len(seasons) > 1 and not quiet:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            for season_year in sorted(season_accuracies.keys()):
                print(f"{season_year}: {season_accuracies[season_year]:.2f}%")
            avg_accuracy = sum(season_accuracies.values()) / len(season_accuracies)
            print(f"\nAverage Accuracy: {avg_accuracy:.2f}%")
            print(f"{'='*70}")

        # Return the most recent season's accuracy or average if multiple
        if len(seasons) == 1:
            return season_accuracies.get(seasons[0], 0)
        else:
            return (
                sum(season_accuracies.values()) / len(season_accuracies)
                if season_accuracies
                else 0
            )

    return 0


def get_past_predictions_model(model, spread_line=False):
    """Get week-by-week 2025 results with consistency metrics."""
    current_week = nfl.get_current_week()

    if spread_line:
        print("Using nflreadpy spread line for predictions.")

    game_to_predict = run_query(
        f"SELECT * FROM training_data td where td.season == 2025 and td.week <= {current_week} AND (away_score IS NOT NULL OR home_score IS NOT NULL) order by week asc"
    )
    games_df = pd.DataFrame(game_to_predict)

    correct = 0
    total = 0
    week = 1

    overall_correct = 0
    overall_correct_total = 0

    predictions = predict(games_df, model, spread_line)

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


def predict(games_df, model, spread_line=False):

    predictions = []

    for game in games_df.iterrows():
        features = model["features"]
        season = game[1]["season"]
        week = game[1]["week"]
        if spread_line:
            spread = game[1]["spread_line"]
            spread = -spread
        else:
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

        cover_spread = float(
            cover_spread_by(game[1][fav], abs_spread, predicted_winner, pidicted_diff),
        )

        game_data = dict(game[1])
        game_data["spread"] = abs_spread
        game_data["cover_spread_by"] = cover_spread
        game_data["predicted_diff"] = float(pidicted_diff)
        confidence_score = get_confidence_score(game_data)

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
            "cover_spread_by": cover_spread,
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
            "confidence_score": confidence_score,
        }
        predictions.append(prediction)

    sorted_predictions = sorted(
        predictions, key=lambda x: (x["week"], x["confidence_score"]), reverse=False
    )
    for week, games in groupby(sorted_predictions, key=lambda x: x["week"]):
        games_list = list(games)
        for i, game in enumerate(games_list):
            game["confidence"] = i + 1

    return predictions
