from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, DateTime, Integer, String, Float,
    UniqueConstraint, Index
)
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base

def utcnow():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(30), default="viewer", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    feed: Mapped[str] = mapped_column(String(50), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "source", "feed", "source_record_id",
            name="uq_sensor_source_feed_record"
        ),
        Index("ix_sensor_readings_recorded_at", "recorded_at"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    parameter: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=True)
