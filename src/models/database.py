"""SQLAlchemy Core database connection management and schema migrations."""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text

from src.config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

# Singleton engine
_engine = None


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.environ.get("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # SQLite pragmas applied on every new raw DBAPI connection
    if db_url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return _engine


@contextmanager
def get_db():
    """Context manager yielding a SQLAlchemy Connection with auto-commit/rollback."""
    engine = get_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except Exception:
            trans.rollback()
            raise


def row_to_dict(row) -> dict:
    """Convert a SQLAlchemy Row to a plain dict."""
    return dict(row._mapping)


def get_current_version(conn) -> int:
    """Get the current schema version."""
    try:
        result = conn.execute(text("SELECT MAX(version) FROM schema_version"))
        row = result.fetchone()
        return row[0] if row[0] is not None else 0
    except Exception:
        return 0


def initialize_database():
    """Initialize the database and run any pending migrations."""
    from src.models.migrations import MIGRATIONS
    from src.models.fixups import fix_truncated_output_folders
    from src.models.settings import insert_default_settings

    os.makedirs(os.path.dirname(SQLITE_DB_PATH) if os.path.dirname(SQLITE_DB_PATH) else ".", exist_ok=True)

    with get_db() as conn:
        current_version = get_current_version(conn)
        logger.info(f"Current database schema version: {current_version}")

        for version, description, statements in MIGRATIONS:
            if version > current_version:
                logger.info(f"Applying migration {version}: {description}")
                for sql in statements:
                    conn.execute(text(sql))
                conn.execute(
                    text("INSERT INTO schema_version (version) VALUES (:v)"),
                    {"v": version}
                )
                logger.info(f"Migration {version} applied successfully")

        # One-time Python fixup for migration 11
        if current_version < 11:
            fix_truncated_output_folders(conn)

        # Insert default settings if they don't exist
        insert_default_settings(conn)
