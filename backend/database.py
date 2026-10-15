# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

import os

from pathlib import Path
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_PATH)


# --------------------------------------------------
# 2. READ DATABASE URL
# --------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. "
        "Please configure it in backend/.env"
    )


# --------------------------------------------------
# 3. CREATE DATABASE ENGINE
# --------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# --------------------------------------------------
# 4. CREATE SESSION FACTORY
# --------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# --------------------------------------------------
# 5. CREATE BASE CLASS
# --------------------------------------------------

Base = declarative_base()


# --------------------------------------------------
# 6. DATABASE SESSION DEPENDENCY
# --------------------------------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()