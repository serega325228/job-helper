from uuid import UUID, uuid4

from infrastructure.models.base import Base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import String, Text, Uuid


class PreferenceIntent(Base):
    __tablename__ = "preference_intents"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        index=True,
    )

    name: Mapped[str] = mapped_column(String(200))

    description: Mapped[str | None] = mapped_column(Text)

    target_titles: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    preferred_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    preferred_industries: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    preferred_companies: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    excluded_titles: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    locations: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    work_formats: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
    )

    min_seniority: Mapped[str | None]
    max_seniority: Mapped[str | None]

    weight: Mapped[float] = mapped_column(default=1.0)
    enabled: Mapped[bool] = mapped_column(default=True)
