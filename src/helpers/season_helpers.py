import nflreadpy as nfl


def get_league_season() -> int:
    """Season currently being played or about to start.

    nfl.get_current_season() does not roll over until the Thursday after Labor
    Day, so through the preseason it still reports the prior season even though
    schedules and betting lines for the new one are already published. The
    league year (rolls over March 15) matches what those sources are serving.
    """
    return nfl.get_current_season(roster=True)


def get_league_week() -> int:
    """Current week within get_league_season()."""
    return nfl.get_current_week(roster=True)
