import nflreadpy as nfl
import requests
from bs4 import BeautifulSoup
import re
from sqlalchemy import text

from src.helpers.database_helpers import get_db_engine


class YahooSpreadClient:
    """Fetch betting spreads from yahoo fantasy football site."""

    def __init__(
        self,
        base_url="https://football.fantasysports.yahoo.com/pickem/pickdistribution",
    ):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10

        self.team_abbr = {
            "Buffalo": "BUF",
            "Houston": "HOU",
            "Chicago": "CHI",
            "Pittsburgh": "PIT",
            "New England": "NE",
            "Cincinnati": "CIN",
            "Detroit": "DET",
            "New York (NYG)": "NYG",
            "Green Bay": "GB",
            "Minnesota": "MIN",
            "Seattle": "SEA",
            "Tennessee": "TEN",
            "Kansas City": "KC",
            "Indianapolis": "IND",
            "Baltimore": "BAL",
            "New York (NYJ)": "NYJ",
            "Las Vegas": "LV",
            "Cleveland": "CLE",
            "Jacksonville": "JAX",
            "Arizona": "ARI",
            "Philadelphia": "PHI",
            "Dallas": "DAL",
            "New Orleans": "NO",
            "Atlanta": "ATL",
            "Los Angeles (LAR)": "LA",
            "Los Angeles (LAC)": "LAC",
            "Tampa Bay": "TB",
            "San Francisco": "SF",
            "Carolina": "CAR",
            "Washington": "WAS",
            "Denver": "DEN",
            "Miami": "MIA",
        }

    def save_to_db(self, game_spreads):
        engine = get_db_engine()
        create_table_query = """
            CREATE TABLE IF NOT EXISTS yahoo_spreads (
            home_team TEXT,
            away_team TEXT,
            week INTEGER,
            season INTEGER,
            yahoo_spread REAL,
            PRIMARY KEY (home_team, away_team, week, season)
            )
        """
        insert_query = text(
            """
            INSERT INTO yahoo_spreads (home_team, away_team, week, season, yahoo_spread)
            VALUES (:home_team, :away_team, :week, :season, :yahoo_spread)
            ON CONFLICT(home_team, away_team, week, season) 
            DO UPDATE SET yahoo_spread = excluded.yahoo_spread
        """
        )

        with engine.connect() as conn:
            conn.execute(text(create_table_query))
            conn.commit()

            conn.execute(insert_query, game_spreads)
            conn.commit()

    def get_all_week_spreads(self):
        all_spreads = []
        current_week = nfl.get_current_week()

        for week in range(5, current_week + 1):
            week_spreads = self.get_week_spread(week)
            all_spreads.extend(week_spreads)
        return all_spreads

    def get_week_spread(self, week):

        url = f"{self.base_url}?gid=&week={week}&type=s"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all Favorite and Underdog tags and pair them
        games = []
        # Only use tags that are exactly 'Favorite' or 'Underdog'
        # Find all <dd class='team'> tags
        team_dds = soup.find_all("dd", class_="team")
        teams = []
        for dd in team_dds:
            team_a = dd.find("a")
            if not team_a:
                continue
            team_name = team_a.get_text(strip=True)
            spread_match = re.search(r"\((-?\d+\.?\d*) pts\)", dd.get_text())
            if not spread_match:
                continue
            spread = float(spread_match.group(1))
            is_home = "@" in dd.get_text()
            teams.append({"team": team_name, "spread": spread, "is_home": is_home})
        # Group every two as a game
        games = []
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):
                break
            t1 = teams[i]
            t2 = teams[i + 1]
            # Determine home/away
            if t1["is_home"]:
                home, away = t1, t2
            elif t2["is_home"]:
                home, away = t2, t1
            else:
                continue
            # The favorite is the team with the negative spread
            if home["spread"] < away["spread"]:
                # Home is favored
                spread = -abs(home["spread"])
            else:
                # Away is favored
                spread = abs(away["spread"])
            home_abbr = self.team_abbr.get(home["team"])
            away_abbr = self.team_abbr.get(away["team"])

            if not home_abbr or not away_abbr:
                print(f"Unknown team abbreviation for {home['team']} or {away['team']}")
                continue

            games.append(
                {
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "week": week,
                    "season": nfl.get_current_season(),
                    "yahoo_spread": spread,
                }
            )
        self.save_to_db(games)
        return games
