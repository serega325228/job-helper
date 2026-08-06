from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, String, Text

from src.infrastructure.models.base import Base


class Profile(Base):

    __tablename__ = "profiles"

    def apply_analysis(
        self,
        *,
        profile_summary: str,
        structured_data: dict,
        preferences: dict,
        target_titles: list[str],
        contacts: dict
    ) -> None:
        self.profile_summary = profile_summary.strip()
        self.structured_data = structured_data
        self.preferences = preferences
        self.target_titles = target_titles
        self.contacts = contacts

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

    analysis_status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        nullable=False,
        index=True,
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
