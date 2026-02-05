import nflreadpy as nfl
import pandas as pd


# Function to find the most recent previous QB for a team
def get_previous_qb(team, current_week, schedule_df):
    # Get all previous games for this team
    previous_games = schedule_df[
        ((schedule_df["home_team"] == team) | (schedule_df["away_team"] == team))
        & (schedule_df["week"] < current_week)
    ].sort_values("week", ascending=False)

    if len(previous_games) == 0:
        return None

    # Get the most recent game
    most_recent = previous_games.iloc[0]

    # Return the QB name based on whether team was home or away
    if most_recent["home_team"] == team:
        return most_recent["home_qb_name"]
    else:
        return most_recent["away_qb_name"]


def get_qb_change():
    """Identify quarterback changes for the current week."""
    # Load the schedule data
    current_season = nfl.get_current_season()
    schedule_df = nfl.load_schedules(seasons=current_season).to_pandas()

    # Filter for regular season games only (except current season)
    schedule_df = schedule_df[
        (schedule_df["season"] == current_season) | (schedule_df["game_type"] == "REG")
    ].copy()

    # Set the current week
    current_week = nfl.get_current_week()

    # Get current week games - unpivot to get each team with their QB
    current_week_home = schedule_df[schedule_df["week"] == current_week][
        ["home_team", "home_qb_name", "week"]
    ].copy()
    current_week_home.columns = ["team", "current_qb", "week"]

    current_week_away = schedule_df[schedule_df["week"] == current_week][
        ["away_team", "away_qb_name", "week"]
    ].copy()
    current_week_away.columns = ["team", "current_qb", "week"]

    current_week_teams = pd.concat(
        [current_week_home, current_week_away], ignore_index=True
    )

    # Get previous QB for each team
    current_week_teams["last_weeks_qb"] = current_week_teams["team"].apply(
        lambda x: get_previous_qb(x, current_week, schedule_df)
    )

    # Filter for QB changes
    qb_changes = current_week_teams[
        (current_week_teams["current_qb"] != current_week_teams["last_weeks_qb"])
        | (current_week_teams["last_weeks_qb"].isna())
    ].copy()

    # Rename and select columns to match SQL output
    qb_changes = qb_changes.rename(
        columns={"week": "current_week", "current_qb": "this_weeks_qb"}
    )

    qb_changes = qb_changes[
        ["team", "current_week", "this_weeks_qb", "last_weeks_qb"]
    ].sort_values("team")

    # Print the results
    print(qb_changes.to_csv(index=False, sep="\t"))
