import nflreadpy as nfl

from src.data.yahoo_spreads import YahooSpreadClient
from src.helpers.database_helpers import get_db_engine
import sqlalchemy


def update_current_spreads():
    """Update current week spread data."""
    current_week = nfl.get_current_week()
    current_season = nfl.get_current_season()

    client = YahooSpreadClient()
    yahoo_spread_data = client.scrape_current_season_spread(current_week)
    spread_line_data = nfl.load_schedules(seasons=[current_season]).to_pandas()
    spread_line_data = spread_line_data[
        (spread_line_data["season"] == current_season)
        & (spread_line_data["week"] == current_week)
    ]

    engine = get_db_engine()

    # Update yahoo_spread data in games table
    with engine.connect() as conn:
        for row in yahoo_spread_data:
            conn.execute(
                sqlalchemy.text(
                    f"""UPDATE games 
                        SET yahoo_spread = {row['yahoo_spread']} 
                        WHERE home_team = '{row['home_team']}' 
                        AND away_team = '{row['away_team']}' 
                        AND week = {current_week} 
                        AND season = {current_season}"""
                )
            )
        conn.commit()

    # Update yahoo_spread data in training_data table
    with engine.connect() as conn:
        for row in yahoo_spread_data:
            conn.execute(
                sqlalchemy.text(
                    f"""UPDATE training_data 
                        SET yahoo_spread = {row['yahoo_spread']} 
                        WHERE home_team = '{row['home_team']}' 
                        AND away_team = '{row['away_team']}' 
                        AND week = {current_week} 
                        AND season = {current_season}"""
                )
            )
        conn.commit()

    # Update yahoo_spread data in yahoo_spreads table
    with engine.connect() as conn:
        for row in yahoo_spread_data:
            conn.execute(
                sqlalchemy.text(
                    f"""UPDATE yahoo_spreads 
                        SET yahoo_spread = {row['yahoo_spread']} 
                        WHERE home_team = '{row['home_team']}' 
                        AND away_team = '{row['away_team']}' 
                        AND week = {current_week} 
                        AND season = {current_season}"""
                )
            )
        conn.commit()

    # Update spread_line data in games table
    with engine.connect() as conn:
        for _, row in spread_line_data.iterrows():
            conn.execute(
                sqlalchemy.text(
                    f"""UPDATE games 
                        SET spread_line = {row['spread_line']} 
                        WHERE game_id = '{row['game_id']}'"""
                )
            )
        conn.commit()

    # Update spread_line data in training_data table
    with engine.connect() as conn:
        for _, row in spread_line_data.iterrows():
            conn.execute(
                sqlalchemy.text(
                    f"""UPDATE training_data 
                        SET spread_line = {row['spread_line']} 
                        WHERE game_id = '{row['game_id']}'"""
                )
            )
        conn.commit()

    print(f"Updated spread data for Week {current_week}, {current_season}")
    print(f"  - Yahoo spreads updated: {len(yahoo_spread_data)} games")
    print(f"  - Spread lines updated: {len(spread_line_data)} games")
