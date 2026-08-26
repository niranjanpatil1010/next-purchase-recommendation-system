"""
db.py
────────────────────────────────────────────────────────────────
Single shared DB connection point.
Both app.py (health check) and predict_new_user.py (read/write) use this.
"""

import os
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL")

# One engine, reused everywhere (connection pooling handled by SQLAlchemy)
engine = create_engine(DB_URL, pool_pre_ping=True)


def get_engine():
    """Return the shared SQLAlchemy engine."""
    return engine


def check_db_connection():
    """
    Quick health check.
    Returns (True, None) if DB is reachable, else (False, error_message).
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)
