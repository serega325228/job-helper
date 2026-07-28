from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, DateTime, String, Text

from src.infrastructure.models.base import Base


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

    name: Mapped[str] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(
        String(10),
        default="ru",
    )

    resume_type: Mapped[str] = mapped_column(
        String(30),
        default="base",
    )

    content: Mapped[dict] = mapped_column(JSON)

    file_path: Mapped[str | None] = mapped_column(Text)
    file_format: Mapped[str | None] = mapped_column(String(20))

    model_name: Mapped[str | None] = mapped_column(String(255))
    prompt_version: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
