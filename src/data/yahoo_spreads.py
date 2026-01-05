import nflreadpy as nfl
import requests
from bs4 import BeautifulSoup
import re
from sqlalchemy import text
import pandas as pd
from pathlib import Path

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

    def backup_to_csv(self, game_spreads):
        """Backup yahoo spreads to CSV file in yahoo_spread_data directory.
        Stores all spreads for a season in a single file, updating existing records."""
        if not game_spreads:
            return

        # Create directory if it doesn't exist
        backup_dir = Path("yahoo_spread_data")
        backup_dir.mkdir(exist_ok=True)

        # Single file per season
        season = game_spreads[0]["season"]
        filename = backup_dir / f"yahoo_spreads_{season}.csv"

        # Convert new spreads to DataFrame
        new_df = pd.DataFrame(game_spreads)

        # Load existing data if file exists
        if filename.exists():
            existing_df = pd.read_csv(filename)
            # Combine and update (new data overwrites old for same week/teams)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates, keeping last (newest) entry
            combined_df = combined_df.drop_duplicates(
                subset=["home_team", "away_team", "week", "season"], keep="last"
            )
            combined_df = combined_df.sort_values(["week", "home_team"])
        else:
            combined_df = new_df.sort_values(["week", "home_team"])

        # Save to CSV
        combined_df.to_csv(filename, index=False)
        print(f"✓ Backed up {len(game_spreads)} spreads to {filename}")

    def load_all_from_csv(self):
        """Load all spreads from all CSV backup files in the directory."""
        backup_dir = Path("yahoo_spread_data")

        if not backup_dir.exists():
            return []

        all_spreads = []

        # Find all yahoo spread CSV files in the directory
        csv_files = list(backup_dir.glob("yahoo_spreads_*.csv"))

        if not csv_files:
            return []

        # Load data from all season files
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_spreads.extend(df.to_dict("records"))
                print(f"✓ Loaded {len(df)} spreads from {csv_file.name}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

        return all_spreads

    def load_from_csv(self, week):
        """Load spreads from CSV backup for a specific week across all seasons in directory."""
        backup_dir = Path("yahoo_spread_data")

        if not backup_dir.exists():
            return []

        all_week_spreads = []

        # Find all yahoo spread CSV files in the directory
        csv_files = list(backup_dir.glob("yahoo_spreads_*.csv"))

        if not csv_files:
            return []

        # Load data from all season files
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Filter for the specific week
                week_df = df[df["week"] == week]
                if len(week_df) > 0:
                    all_week_spreads.extend(week_df.to_dict("records"))
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

        return all_week_spreads

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
        """Load historical spreads from CSV and scrape current season from website.
        Website data takes precedence over CSV data for matching records."""

        # First, load all historical data from CSV files
        print("Loading historical spread data from CSV files...")
        csv_spreads = self.load_all_from_csv()

        # Convert to DataFrame for easier merging
        if csv_spreads:
            csv_df = pd.DataFrame(csv_spreads)
        else:
            csv_df = pd.DataFrame(
                columns=["home_team", "away_team", "week", "season", "yahoo_spread"]
            )

        # Now scrape website for current season (accumulate without saving)
        print(f"Scraping website for current season spreads...")
        current_season = nfl.get_current_season()
        current_week = nfl.get_current_week()
        website_spreads = []

        for week in range(1, current_week + 1):
            week_spreads = self.scrape_current_season_spread(week)
            website_spreads.extend(week_spreads)

        # Convert website data to DataFrame
        if website_spreads:
            website_df = pd.DataFrame(website_spreads)

            # Remove current season data from CSV (will be replaced by website data)
            csv_df = csv_df[csv_df["season"] != current_season]

            # Combine: CSV historical data + website current season data
            combined_df = pd.concat([csv_df, website_df], ignore_index=True)
        else:
            # No website data, use CSV only
            combined_df = csv_df

        # Remove any duplicates (shouldn't be any, but just in case)
        combined_df = combined_df.drop_duplicates(
            subset=["home_team", "away_team", "week", "season"], keep="last"
        )

        # Convert back to list of dicts
        all_spreads = combined_df.to_dict("records")

        # Save all spreads to database at once
        if all_spreads:
            print(f"Saving {len(all_spreads)} spreads to database...")
            self.save_to_db(all_spreads)

        # Backup to CSV - group by season and save each season separately
        if len(combined_df) > 0:
            for season in combined_df["season"].unique():
                season_data = combined_df[combined_df["season"] == season].to_dict(
                    "records"
                )
                self.backup_to_csv(season_data)

        print(f"Total spreads loaded: {len(all_spreads)}")
        return all_spreads

    def scrape_current_season_spread(self, week):
        """Scrape current seasons spreads for a specific week from Yahoo website."""

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

        return games
