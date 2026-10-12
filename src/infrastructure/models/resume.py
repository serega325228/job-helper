from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, String, Text

from src.infrastructure.models.base import Base


class ResumeStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


class ResumeType(StrEnum):
    MASTER = "master"
    TAILORED = "tailored"


class ResumeLanguage(StrEnum):
    RU = "ru"
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.id"),
    )

    vacancy_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("vacancies.id"),
        nullable=True,
    )

    vacancy_match_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("vacancy_matches.id"),
        nullable=True,
    )

    title: Mapped[str | None] = mapped_column(String, nullable=True)
    language: Mapped[str] = mapped_column(
        String(10),
        default=ResumeLanguage.EN,
    )

    type: Mapped[str] = mapped_column(
        String(30),
        default=ResumeType.TAILORED,
    )

    raw_data: Mapped[dict] = mapped_column(Text)
    processed_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    filename: Mapped[str | None] = mapped_column(String, nullable=True)
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

    status: Mapped[str] = mapped_column(String, default=ResumeStatus.PENDING)

    # cover_letter: Mapped[str | None] = mapped_column(Text, nullable=True)
    # outreach_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # interview_prep: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index(
            "uq_resumes_profile_master_resume",
            "profile_id",
            unique=True,
            where=text("type = 'master'"),
        ),
    )
