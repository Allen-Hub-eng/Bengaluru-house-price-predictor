import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "backend" / ".env"

# Load local environment variables.
# On Render, environment variables are configured in the dashboard.
load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
TIDB_CA_PATH = os.getenv("TIDB_CA_PATH")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")

# Configure TLS certificate verification for TiDB Cloud.
connect_args = {}

if TIDB_CA_PATH:
    if not Path(TIDB_CA_PATH).is_file():
        raise RuntimeError(
            f"TiDB CA certificate not found: {TIDB_CA_PATH}"
        )

    connect_args["ssl"] = {
        "ca": TIDB_CA_PATH
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=2,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()