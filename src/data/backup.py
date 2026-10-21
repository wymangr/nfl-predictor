import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path


def backup_database(db_path="nfl-prediction.db", backup_dir="backups"):
    """
    Create a consistent SQLite snapshot, including committed WAL data.

    Args:
        db_path: Path to the database file (default: "nfl-prediction.db")
        backup_dir: Directory to store backups (default: "backups")
            Relative paths use the working directory, like the database engine.

    Returns:
        str: Path to the backup file

    Raises:
        FileNotFoundError: If the database file doesn't exist
        OSError: If backup creation fails
        sqlite3.Error: If the database cannot be backed up or verified
    """
    # Back up the same database that the refresh will open, including when
    # running against an isolated copy from another working directory.
    source_db = Path(db_path).resolve()
    backup_folder = Path(backup_dir).resolve()

    # Check if source database exists
    if not source_db.is_file():
        raise FileNotFoundError(f"Database file not found: {source_db}")

    # Create backup directory if it doesn't exist
    backup_folder.mkdir(parents=True, exist_ok=True)

    # Never overwrite an earlier backup, even if two calls share a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_filename = f"nfl-prediction_backup_{timestamp}.db"
    backup_path = backup_folder / backup_filename
    backup_path.touch(exist_ok=False)

    try:
        with closing(
            sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)
        ) as source, closing(sqlite3.connect(backup_path)) as target:
            source.backup(target)
            if target.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise sqlite3.DatabaseError("Database backup failed its integrity check")
    except BaseException:
        # Do not leave an incomplete snapshot that looks like a usable backup.
        backup_path.unlink(missing_ok=True)
        raise

    print(f"✅ Database backup created successfully:")
    print(f"   Source: {source_db}")
    print(f"   Backup: {backup_path}")
    print(f"   Size: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")

    return str(backup_path)
