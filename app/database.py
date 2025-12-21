from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import sqlite3

SQLALCHAMY_DATABASE_URL = 'sqlite:///./rag.db'

# Standard SQLite configuration for FastAPI
engine = create_engine(
    SQLALCHAMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30
    },
    poolclass=NullPool
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

def get_db():
    """
    FastAPI dependency for database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"❌ Database error: {e}")
        db.rollback()
        raise
    finally:
        db.commit()
        db.close()