import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Provides a database session for the duration of one request, closing it afterward.

    Parameters:
    - none

    Returns:
    - Session: a SQLAlchemy session, yielded to FastAPI's dependency injection — used by services, closed automatically when the request finishes
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()