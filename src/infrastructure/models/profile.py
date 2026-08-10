from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime, String, Text

from src.infrastructure.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.models.preference_intent import PreferenceIntent
    from src.infrastructure.models.vacancy_match import VacancyMatch


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str | None] = mapped_column(String(255))
    raw_story: Mapped[str] = mapped_column(Text, default="", nullable=False)
    profile_summary: Mapped[str | None] = mapped_column(Text)
    analysis_status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        nullable=False,
        index=True,
    )

    skills: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    experience: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    education: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    seniority: Mapped[str | None] = mapped_column(String(50), index=True)
    experience_years: Mapped[float | None] = mapped_column(Float)
    contacts: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    preference_intents: Mapped[list[PreferenceIntent]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    vacancy_matches: Mapped[list[VacancyMatch]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def apply_analysis(
        self,
        *,
        profile_summary: str,
        skills: list[str],
        experience: list[str],
        education: list[str],
        seniority: str | None,
        experience_years: float | None,
    ) -> None:
        self.profile_summary = profile_summary.strip()
        self.skills = skills
        self.experience = experience
        self.education = education
        self.seniority = seniority
        self.experience_years = experience_years
