import os
import shutil
from datetime import datetime
from pathlib import Path


def backup_database(db_path="nfl-prediction.db", backup_dir="backups"):
    """
    Create a backup of the NFL prediction database.

    Args:
        db_path: Path to the database file (default: "nfl-prediction.db")
        backup_dir: Directory to store backups (default: "backups")

    Returns:
        str: Path to the backup file

    Raises:
        FileNotFoundError: If the database file doesn't exist
        OSError: If backup creation fails
    """
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent

    # Construct full paths
    source_db = project_root / db_path
    backup_folder = project_root / backup_dir

    # Check if source database exists
    if not source_db.exists():
        raise FileNotFoundError(f"Database file not found: {source_db}")

    # Create backup directory if it doesn't exist
    backup_folder.mkdir(exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"nfl-prediction_backup_{timestamp}.db"
    backup_path = backup_folder / backup_filename

    # Copy the database file
    shutil.copy2(source_db, backup_path)

    print(f"✅ Database backup created successfully:")
    print(f"   Source: {source_db}")
    print(f"   Backup: {backup_path}")
    print(f"   Size: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")

    return str(backup_path)
