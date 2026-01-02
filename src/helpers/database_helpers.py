from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Any
import sqlalchemy


def get_db_engine(db_path: str = "nfl-prediction.db") -> Engine:
    """Create a SQLAlchemy engine for the SQLite DB."""
    engine = create_engine(f"sqlite:///{db_path}")
    return engine


def run_query(sql: str, db_path: str = "nfl-prediction.db") -> Any:
    """Run a SQL query and return results as a list of dicts."""
    engine = get_db_engine(db_path)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(sql))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return rows
