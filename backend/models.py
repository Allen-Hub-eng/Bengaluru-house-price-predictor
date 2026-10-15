# --------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Numeric,
    func,
)

from sqlalchemy.orm import relationship

from backend.database import Base


# --------------------------------------------------
# 1. USERS TABLE
# --------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    username = Column(
        String(50),
        nullable=False,
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    password_hash = Column(
        String(255),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        server_default=func.current_timestamp(),
        nullable=False,
    )

    predictions = relationship(
        "PredictionHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# --------------------------------------------------
# 2. PREDICTION HISTORY TABLE
# --------------------------------------------------

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    location = Column(
        String(150),
        nullable=False,
    )

    total_sqft = Column(
        Numeric(10, 2),
        nullable=False,
    )

    bhk = Column(
        Integer,
        nullable=False,
    )

    bath = Column(
        Integer,
        nullable=False,
    )

    balcony = Column(
        Integer,
        nullable=True,
    )

    predicted_price_lakhs = Column(
        Numeric(12, 2),
        nullable=False,
    )

    predicted_price_crores = Column(
        Numeric(12, 4),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        server_default=func.current_timestamp(),
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="predictions",
    )