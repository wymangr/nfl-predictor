"""Current-week nflreadpy game data and Yahoo lines, without predicting again."""

import time

import nflreadpy as nfl
import pandas as pd
from sqlalchemy import text

from src.data.yahoo_spreads import YahooSpreadClient
from src.helpers.database_helpers import get_db_engine
from src.helpers.season_helpers import get_league_season, get_league_week
from src.model.predict import check_correct_prediction


def _columns(conn, table):
    return {row[1]: row[2] for row in conn.execute(text(f'PRAGMA table_info("{table}")'))}


def _quoted(column):
    return '"' + column.replace('"', '""') + '"'


def _grade(pick, home_score, away_score):
    # Use the spread recorded with the pick, never today's market line.
    spread = pick["spread"]
    if spread is None:
        raise ValueError(f"Saved pick {pick['game_id']} has no spread; refresh cancelled.")
    signed_spread = -abs(spread) if pick["spread_favorite"] == pick["home_team"] else abs(spread)
    result = check_correct_prediction(
        signed_spread, home_score, away_score,
        pick["home_team"], pick["away_team"], pick["predicted_winner"],
    )
    return "push" if result == "push" else str(int(result))


def _archive_saved_game(conn, game):
    """Grade/move a saved pick only; never run the model or change pick fields."""
    scope = {key: game[key] for key in ("game_id", "season", "week")}
    where = "game_id = :game_id AND season = :season AND week = :week"
    future_cols = _columns(conn, "future_predictions")
    past_cols = _columns(conn, "past_predictions")
    past = (conn.execute(text(f"SELECT * FROM past_predictions WHERE {where}"), scope)
            .mappings().all() if past_cols else [])
    future = (conn.execute(text(f"SELECT * FROM future_predictions WHERE {where}"), scope)
              .mappings().all() if future_cols else [])
    if len(past) > 1 or len(future) > 1:
        raise ValueError(f"Duplicate saved picks for {scope['game_id']}; refresh cancelled.")
    if past:
        conn.execute(text(f"UPDATE past_predictions SET correct = :correct WHERE {where}"),
                     {**scope, "correct": _grade(past[0], game["home_score"], game["away_score"])})
        outcome = "already_past"
    elif future:
        if not past_cols:
            definitions = ", ".join(
                f'{_quoted(col)} {"TEXT" if col == "correct" else dtype}'
                for col, dtype in future_cols.items())
            conn.execute(text(f"CREATE TABLE past_predictions ({definitions})"))
            past_cols = _columns(conn, "past_predictions")
        if not future_cols.keys() <= past_cols.keys():
            raise ValueError("Prediction schemas differ; refresh cancelled to avoid losing pick fields.")
        pick = dict(future[0])
        pick["correct"] = _grade(pick, game["home_score"], game["away_score"])
        cols = ", ".join(_quoted(col) for col in pick)
        params = ", ".join(f":p{i}" for i in range(len(pick)))
        conn.execute(text(f"INSERT INTO past_predictions ({cols}) VALUES ({params})"),
                     {f"p{i}": val for i, val in enumerate(pick.values())})
        outcome = "archived"
    else:
        outcome = "completed_without_pick"
    if future:
        conn.execute(text(f"DELETE FROM future_predictions WHERE {where}"), scope)
    return outcome


def _reconcile_completed_predictions(conn):
    """Catch completed saved picks from any week, including a Monday rollover."""
    counts = {"archived": 0, "already_past": 0}
    if not _columns(conn, "future_predictions") or not _columns(conn, "games"):
        return counts
    games = conn.execute(text("""
        SELECT g.* FROM games g WHERE g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL AND EXISTS (
            SELECT 1 FROM future_predictions fp WHERE fp.game_id = g.game_id
            AND fp.season = g.season AND fp.week = g.week)
    """)).mappings().all()
    for game in games:
        counts[_archive_saved_game(conn, game)] += 1
    return counts


def reconcile_completed_predictions():
    """Reconcile saved picks with local results/QBs; no downloads or model runs."""
    engine = get_db_engine()
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            counts = _reconcile_completed_predictions(conn)
            counts.update(_sync_saved_metadata(conn))
    finally:
        engine.dispose()
    print(f"Saved predictions reconciled: {counts['archived']} moved to past; "
          f"{counts['already_past']} already in past; "
          f"{counts['results_updated']} results corrected; "
          f"{counts['qb_warnings_updated']} QB warnings updated. Original picks preserved.")
    return counts


def _qb_flags(conn, game, season, week):
    flags = {}
    for side in ("home", "away"):
        current = game.get(f"{side}_qb_name")
        if not current:
            continue
        previous = conn.execute(text("""
            SELECT qb FROM (
                SELECT week, home_qb_name AS qb FROM games
                WHERE season = :season AND week < :week AND home_team = :team
                UNION ALL
                SELECT week, away_qb_name AS qb FROM games
                WHERE season = :season AND week < :week AND away_team = :team
            ) WHERE qb IS NOT NULL AND qb != '' ORDER BY week DESC LIMIT 1
        """), {"season": season, "week": week, "team": game[f"{side}_team"]}).scalar()
        if previous:
            flags[f"{side}_qb_changed"] = "Y" if current != previous else "N"
    return flags


def _sync_saved_metadata(conn):
    """A data refresh may change outcomes/QBs, never the original model pick."""
    counts = {"results_updated": 0, "qb_warnings_updated": 0}
    games_cols = _columns(conn, "games")
    if not games_cols:
        return counts
    if _columns(conn, "past_predictions"):
        rows = conn.execute(text("""
            SELECT pp.*, g.home_score AS final_home, g.away_score AS final_away
            FROM past_predictions pp JOIN games g ON g.game_id = pp.game_id
            AND g.season = pp.season AND g.week = pp.week
            WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        """)).mappings().all()
        for pick in rows:
            correct = _grade(pick, pick["final_home"], pick["final_away"])
            if str(pick["correct"]) != correct:
                conn.execute(text("""UPDATE past_predictions SET correct = :correct
                    WHERE game_id = :game_id AND season = :season AND week = :week"""),
                    {"game_id": pick["game_id"], "season": pick["season"],
                     "week": pick["week"], "correct": correct})
                counts["results_updated"] += 1

    future_cols = _columns(conn, "future_predictions")
    if future_cols and {"home_qb_name", "away_qb_name"} <= games_cols.keys():
        games = conn.execute(text("""
            SELECT g.* FROM games g WHERE (g.home_score IS NULL OR g.away_score IS NULL)
            AND EXISTS (SELECT 1 FROM future_predictions fp WHERE fp.game_id = g.game_id
                AND fp.season = g.season AND fp.week = g.week)
        """)).mappings().all()
        for game in games:
            flags = {key: value for key, value in _qb_flags(conn, game, game["season"], game["week"]).items()
                     if key in future_cols}
            if not flags:
                continue
            setters = ", ".join(f'{_quoted(col)} = :{col}' for col in flags)
            changed = " OR ".join(f'{_quoted(col)} IS NOT :{col}' for col in flags)
            result = conn.execute(text(f"""UPDATE future_predictions SET {setters}
                WHERE game_id = :game_id AND season = :season AND week = :week
                AND ({changed})"""),
                {"game_id": game["game_id"], "season": game["season"], "week": game["week"], **flags})
            counts["qb_warnings_updated"] += result.rowcount
    return counts


def _catch_up_saved_scores(conn, season_schedule, season, week):
    """Finish outstanding saved games when nflreadpy has already advanced weeks.

    The season schedule was already downloaded. Only finalized games still in
    the saved future slate get results/QBs updated; unrelated history and old
    spreads/features are deliberately left alone.
    """
    if not _columns(conn, "future_predictions"):
        return 0
    outstanding = conn.execute(text("""
        SELECT g.game_id FROM games g JOIN future_predictions fp
        ON fp.game_id = g.game_id AND fp.season = g.season AND fp.week = g.week
        WHERE g.season = :season AND g.week < :week
        AND (g.home_score IS NULL OR g.away_score IS NULL)
    """), {"season": season, "week": week}).scalars().all()
    updates = season_schedule[season_schedule["game_id"].isin(outstanding)]
    if updates["game_id"].duplicated().any():
        raise ValueError("Duplicate outstanding schedule games; refresh cancelled.")
    count = 0
    games_cols = _columns(conn, "games")
    for _, row in updates.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        scores = {"home_score": float(row["home_score"]), "away_score": float(row["away_score"])}
        scope = {"game_id": row["game_id"], "season": int(row["season"]), "week": int(row["week"])}
        qbs = {col: row[col] for col in ("home_qb_name", "away_qb_name", "home_qb_id", "away_qb_id")
               if col in row and col in games_cols and pd.notna(row[col]) and row[col] != ""}
        setters = ", ".join(f'{_quoted(col)} = :{col}' for col in {**scores, **qbs})
        where = "game_id=:game_id AND season=:season AND week=:week"
        conn.execute(text(f"UPDATE games SET {setters} WHERE {where}"), {**scope, **scores, **qbs})
        conn.execute(text(f"UPDATE training_data SET home_score=:home_score, away_score=:away_score WHERE {where}"),
                     {**scope, **scores})
        count += 1
    return count


def refresh_current_week():
    """Update this week's game data/lines and grade existing picks atomically.

    nflreadpy downloads the schedules dataset and filters it by season (its API
    has no week-only download). Update the current week, plus final scores for
    outstanding saved games from the week that just ended. Does not
    touch model files, team statistics, rankings, or derived training features.
    No prediction is invented for a completed game without a saved pick.
    """
    print("Fetching fresh schedule data from nflreadpy...")
    download_started = time.monotonic()
    nfl.clear_cache()
    season, week = get_league_season(), get_league_week()
    # get_league_week() also loads schedules, so include it in the fetch timer.
    schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    download_seconds = round(time.monotonic() - download_started, 2)
    required = {"game_id", "season", "week", "home_score", "away_score"}
    if not required.issubset(schedule.columns):
        raise ValueError("The schedule is missing required fields; no data was changed.")
    season_schedule = schedule[schedule["season"] == season]
    schedule = season_schedule[season_schedule["week"] == week]
    if schedule.empty:
        raise ValueError(f"No schedule found for {season}, week {week}; no data was changed.")
    if schedule["game_id"].isna().any() or schedule["game_id"].duplicated().any():
        raise ValueError("The schedule has missing/duplicate game IDs; no data was changed.")

    warnings = []
    yahoo = YahooSpreadClient()
    try:
        print(f"Fetching Yahoo Pick'em spreads for week {week} only...")
        yahoo_rows = yahoo.scrape_current_season_spread(week)
    except Exception as exc:
        yahoo_rows = []
        warnings.append(f"Yahoo spreads unavailable ({type(exc).__name__}); previous Yahoo lines kept.")
    valid_matchups = set(zip(schedule["home_team"], schedule["away_team"]))
    yahoo_rows = [row for row in yahoo_rows
                  if row.get("season") == season and row.get("week") == week
                  and (row.get("home_team"), row.get("away_team")) in valid_matchups
                  and pd.notna(row.get("yahoo_spread"))]
    yahoo_by_game = {(row["home_team"], row["away_team"]): row for row in yahoo_rows}
    if len(yahoo_by_game) != len(yahoo_rows):
        raise ValueError("Duplicate Yahoo spread matchups; no data was changed.")
    if not yahoo_rows and not warnings:
        warnings.append("Yahoo returned no matching current-week lines; previous Yahoo lines kept.")

    engine = get_db_engine()
    counts = {
        "season": season, "week": week, "source": "nflreadpy",
        "download_seconds": download_seconds,
        "games_checked": 0, "games_updated": 0, "qb_games_updated": 0,
        "completed": 0, "pending": 0, "already_past": 0, "archived": 0,
        "completed_without_pick": 0,
        "market_lines_updated": 0, "yahoo_lines_updated": 0,
        "yahoo_lines_fetched": len(yahoo_rows), "warnings": warnings,
        "archived_other_weeks": 0, "caught_up_games": 0,
    }
    try:
        with engine.begin() as conn:
            # SQLite otherwise auto-commits DDL before the first data UPDATE.
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            games_cols = _columns(conn, "games")
            training_cols = _columns(conn, "training_data")
            if not games_cols or not training_cols:
                raise ValueError("Run a Full Refresh first to populate games and training data.")
            future_cols = _columns(conn, "future_predictions")
            past_cols = _columns(conn, "past_predictions")
            # Refresh raw game metadata just as Full Refresh would, but protect
            # identity and handle paired scores/Yahoo separately. Derived model
            # features are not rebuilt or changed here.
            protected = {"game_id", "season", "week", "home_team", "away_team",
                         "home_score", "away_score", "yahoo_spread"}
            metadata = [col for col in schedule.columns
                        if col in games_cols and col not in protected]
            if yahoo_rows:
                conn.execute(text("""CREATE TABLE IF NOT EXISTS yahoo_spreads (
                    home_team TEXT, away_team TEXT, week INTEGER, season INTEGER,
                    yahoo_spread REAL, game_order INTEGER,
                    PRIMARY KEY (home_team, away_team, week, season))"""))
                if "game_order" not in _columns(conn, "yahoo_spreads"):
                    conn.execute(text("ALTER TABLE yahoo_spreads ADD COLUMN game_order INTEGER"))

            for _, upstream in schedule.iterrows():
                scope = {"game_id": upstream["game_id"], "season": season, "week": week}
                where = "game_id = :game_id AND season = :season AND week = :week"
                game = conn.execute(text(f"SELECT * FROM games WHERE {where}"), scope).mappings().first()
                if game is None:
                    raise ValueError(f"Game {scope['game_id']} is missing locally; run a Full Refresh first.")
                updates = {col: upstream[col] for col in metadata
                           if col in schedule.columns and col in games_cols
                           and pd.notna(upstream[col]) and upstream[col] != ""}
                # Never erase final scores with a stale/partial upstream result.
                if pd.notna(upstream["home_score"]) and pd.notna(upstream["away_score"]):
                    updates.update(home_score=float(upstream["home_score"]),
                                   away_score=float(upstream["away_score"]))
                yahoo_row = yahoo_by_game.get((game["home_team"], game["away_team"]))
                if yahoo_row and "yahoo_spread" in games_cols:
                    updates["yahoo_spread"] = float(yahoo_row["yahoo_spread"])
                # numpy integer scalars otherwise become SQLite BLOBs.
                updates = {col: value.item() if hasattr(value, "item") else value
                           for col, value in updates.items()}
                updates = {col: value for col, value in updates.items()
                           if game[col] != value}
                if updates:
                    setters = ", ".join(f'{_quoted(col)} = :{col}' for col in updates)
                    conn.execute(text(f"UPDATE games SET {setters} WHERE {where}"), {**scope, **updates})
                    counts["games_updated"] += 1
                    if any("_qb_" in col for col in updates):
                        counts["qb_games_updated"] += 1
                    if "spread_line" in updates:
                        counts["market_lines_updated"] += 1
                    if "yahoo_spread" in updates:
                        counts["yahoo_lines_updated"] += 1
                counts["games_checked"] += 1
                game = dict(conn.execute(text(f"SELECT * FROM games WHERE {where}"), scope).mappings().one())
                # Raw current market lines belong in training_data too, so an
                # explicit later Regenerate can use them. Stored picks stay fixed.
                lines = {}
                if "spread_line" in schedule.columns and "spread_line" in training_cols and pd.notna(upstream["spread_line"]):
                    lines["spread_line"] = float(upstream["spread_line"])
                if yahoo_row and "yahoo_spread" in training_cols:
                    lines["yahoo_spread"] = float(yahoo_row["yahoo_spread"])
                if lines:
                    setters = ", ".join(f'{_quoted(col)} = :{col}' for col in lines)
                    conn.execute(text(f"UPDATE training_data SET {setters} WHERE {where}"), {**scope, **lines})
                if yahoo_row:
                    matchup = {"home_team": game["home_team"], "away_team": game["away_team"],
                               "season": season, "week": week}
                    match_where = "home_team=:home_team AND away_team=:away_team AND season=:season AND week=:week"
                    values = {**matchup, "yahoo_spread": float(yahoo_row["yahoo_spread"]),
                              "game_order": yahoo_row.get("game_order")}
                    exists = conn.execute(text(f"SELECT 1 FROM yahoo_spreads WHERE {match_where}"), matchup).first()
                    if exists:
                        conn.execute(text(f"UPDATE yahoo_spreads SET yahoo_spread=:yahoo_spread, "
                                          f"game_order=COALESCE(:game_order,game_order) WHERE {match_where}"), values)
                    else:
                        conn.execute(text("INSERT INTO yahoo_spreads (home_team, away_team, season, week, yahoo_spread, game_order) "
                                          "VALUES (:home_team,:away_team,:season,:week,:yahoo_spread,:game_order)"), values)
                completed = game["home_score"] is not None and game["away_score"] is not None
                if completed:
                    counts["completed"] += 1
                    conn.execute(text(f"""UPDATE training_data SET home_score = :home_score,
                        away_score = :away_score WHERE {where}"""),
                        {**scope, "home_score": game["home_score"], "away_score": game["away_score"]})

                flags = _qb_flags(conn, game, season, week)
                for table, columns in (("future_predictions", future_cols), ("past_predictions", past_cols)):
                    available = {key: val for key, val in flags.items() if key in columns}
                    if available:
                        setters = ", ".join(f'{_quoted(col)} = :{col}' for col in available)
                        conn.execute(text(f"UPDATE {table} SET {setters} WHERE {where}"), {**scope, **available})
                if not completed:
                    counts["pending"] += 1
                    continue

                outcome = _archive_saved_game(conn, game)
                counts[outcome] += 1
                past_cols = _columns(conn, "past_predictions")

            # get_league_week() uses the newly downloaded results, not our DB.
            # Catch Monday's final result before reconciling the previous slate.
            counts["caught_up_games"] = _catch_up_saved_scores(conn, season_schedule, season, week)
            other = _reconcile_completed_predictions(conn)
            counts["archived_other_weeks"] = other["archived"]
    finally:
        engine.dispose()

    if yahoo_rows:
        try:
            yahoo.backup_to_csv(yahoo_rows)
        except Exception as exc:
            warnings.append(f"Database updated, but Yahoo CSV backup failed ({type(exc).__name__}).")

    counts["message"] = (
        f"Week {week}, {season}: fetched fresh nflreadpy schedules in {download_seconds}s; "
        f"checked {counts['games_checked']} games, changed {counts['games_updated']} "
        f"({counts['qb_games_updated']} with QB changes). "
        f"Spread changes: {counts['market_lines_updated']} nflreadpy, "
        f"{counts['yahoo_lines_updated']} Yahoo ({counts['yahoo_lines_fetched']} Yahoo lines fetched). "
        f"{counts['completed']} completed: {counts['archived']} saved picks moved to past, "
        f"{counts['already_past']} already in past; {counts['pending']} games pending. "
        "Picks preserved."
    )
    if counts["completed_without_pick"]:
        counts["message"] += f" {counts['completed_without_pick']} completed games had no saved pick."
    if counts["archived_other_weeks"]:
        counts["message"] += f" Also archived {counts['archived_other_weeks']} completed saved picks from earlier weeks."
    if counts["caught_up_games"]:
        counts["message"] += f" Caught up final scores for {counts['caught_up_games']} outstanding saved games."
    if warnings:
        counts["message"] += " " + " ".join(warnings)
    print(counts["message"])
    return counts