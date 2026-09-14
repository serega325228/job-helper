from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, DateTime, String, Text

from src.infrastructure.models.base import Base


class ResumeProcessingStatus(StrEnum):
    PENDING = "pending"


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

    resume_type: Mapped[str] = mapped_column(
        String(30),
        default=ResumeType.TAILORED,
    )

    content: Mapped[dict] = mapped_column(JSON)

    filename: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    parent_id: Mapped[UUID | None] = mapped_column(nullable=True)
    processed_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    processing_status: Mapped[str] = mapped_column(
        String, default=ResumeProcessingStatus.PENDING
    )
    cover_letter: Mapped[str | None] = mapped_column(Text, nullable=True)
    outreach_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    interview_prep: Mapped[str | None] = mapped_column(Text, nullable=True)
