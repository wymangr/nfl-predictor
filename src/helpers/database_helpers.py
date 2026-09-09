import re

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Any
import sqlalchemy


def get_db_engine(db_path: str = "nfl-prediction.db") -> Engine:
    """Create a SQLAlchemy engine for the SQLite DB."""
    engine = create_engine(f"sqlite:///{db_path}")
    return engine


def _sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "TEXT"


def add_missing_columns(engine: Engine, table_name: str, df: pd.DataFrame) -> None:
    """Add columns present in `df` but missing from an existing table.

    Upstream nflverse data periodically gains new fields; without this an append
    fails with "table X has no column named Y".
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    with engine.connect() as conn:
        exists = conn.execute(
            sqlalchemy.text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=:t"
            ),
            {"t": table_name},
        ).fetchone()
        if not exists:
            return

        existing = {
            row[1]
            for row in conn.execute(sqlalchemy.text(f"PRAGMA table_info({table_name})"))
        }
        missing = [col for col in df.columns if col not in existing]
        if not missing:
            return

        for col in missing:
            conn.execute(
                sqlalchemy.text(
                    f'ALTER TABLE {table_name} ADD COLUMN "{col}" {_sqlite_type(df[col].dtype)}'
                )
            )
        conn.commit()

    print(
        f"⚠️  Schema change detected in {table_name} - added {len(missing)} new column(s): "
        f"{', '.join(missing)}"
    )


def run_query(sql: str, db_path: str = "nfl-prediction.db") -> Any:
    """Run a SQL query and return results as a list of dicts."""
    engine = get_db_engine(db_path)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(sql))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return rows
