from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import sqlite3

SQLALCHAMY_DATABASE_URL = 'sqlite:///./rag.db'

# Configure SQLite with NullPool (no connection pooling) to prevent locking issues
engine = create_engine(
    SQLALCHAMY_DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30  # 30 second timeout for database locks
    },
    poolclass=NullPool  # Disable connection pooling for SQLite to avoid lock issues
)

# Enable pragmas for SQLite to reduce locking
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite pragmas for better concurrency"""
    if isinstance(dbapi_conn, sqlite3.Connection):
        try:
            cursor = dbapi_conn.cursor()
            # Set timeout without WAL mode (NullPool handles connection isolation)
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
            cursor.close()
        except sqlite3.OperationalError:
            # Silently ignore if pragma fails - NullPool still isolates connections
            pass

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()