from __future__ import annotations

import sqlite3
from pathlib import Path


def inspect_database(database_path: Path) -> dict[str, int | str]:
    """Inspect a COLMAP SQLite database and return table row counts."""
    if not database_path.exists():
        return {"error": "database not found"}

    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        counts: dict[str, int | str] = {}
        for table in ["cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = "missing"
        conn.close()
        return counts
    except Exception as exc:
        return {"error": str(exc)}
