from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import Float, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Boolean, DateTime, String, Text

from src.infrastructure.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.models.preference_intent import PreferenceIntent
    from src.infrastructure.models.profile import Profile
    from src.infrastructure.models.vacancy import Vacancy


class VacancyMatch(Base):
    __tablename__ = "vacancy_matches"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vacancy_id: Mapped[UUID] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    preference_intent_id: Mapped[UUID] = mapped_column(
        ForeignKey("preference_intents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    structured_profile_score: Mapped[float] = mapped_column(Float, nullable=False)
    structured_preference_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    profile_rerank_score: Mapped[float | None] = mapped_column(Float)
    preference_rerank_score: Mapped[float | None] = mapped_column(Float)
    total_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    hard_constraints_passed: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    component_scores: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    matched_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    missing_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    explanation: Mapped[str | None] = mapped_column(Text)
    matcher_version: Mapped[str] = mapped_column(String(100), nullable=False)
    reranker_model: Mapped[str | None] = mapped_column(String(255))

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

    profile: Mapped[Profile] = relationship(back_populates="vacancy_matches")
    vacancy: Mapped[Vacancy] = relationship(back_populates="vacancy_matches")
    preference_intent: Mapped[PreferenceIntent] = relationship(
        back_populates="vacancy_matches",
    )

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "vacancy_id",
            name="uq_vacancy_matches_profile_vacancy",
        ),
    )
