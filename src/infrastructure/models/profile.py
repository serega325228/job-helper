from datetime import datetime

from uuid import UUID, uuid4
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.functions import func
from sqlalchemy.types import DateTime, Text, String

from src.infrastructure.models.base import Base


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    raw_story: Mapped[str] = mapped_column(
        Text,
        default="",
    )

    profile_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    structured_data: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    preferences: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    target_titles: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
    )

    contacts: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
