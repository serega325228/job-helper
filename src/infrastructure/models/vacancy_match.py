from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Text, JSON, DateTime, String

from src.infrastructure.models.base import Base


class VacancyMatch(Base):
    __tablename__ = "vacancy_matches"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.id"),
    )

    vacancy_id: Mapped[UUID] = mapped_column(
        ForeignKey("vacancies.id"),
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
    )

    total_score: Mapped[float | None]

    recommendation: Mapped[str | None] = mapped_column(
        String(50),
    )

    hard_constraints_passed: Mapped[bool | None]

    scores: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    matched_skills: Mapped[list[dict]] = mapped_column(
        JSON,
        default=list,
    )

    missing_skills: Mapped[list[dict]] = mapped_column(
        JSON,
        default=list,
    )

    strengths: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
    )

    gaps: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
    )

    risks: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
    )

    explanation: Mapped[str | None] = mapped_column(Text)

    matcher_version: Mapped[str | None] = mapped_column(
        String(100),
    )

    model_name: Mapped[str | None] = mapped_column(
        String(255),
    )

    raw_result: Mapped[dict] = mapped_column(
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

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "vacancy_id",
            name="uq_profile_vacancy_match",
        ),
    )
