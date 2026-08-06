from datetime import datetime

from uuid import UUID, uuid4
from sqlalchemy import JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.functions import func
from sqlalchemy.types import DateTime, Text, String

from src.infrastructure.models.base import Base


class Vacancy(Base):
    __tablename__ = "vacancies"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    company_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("companies.id"),
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        index=True
    )
    external_id: Mapped[str | None] = mapped_column(String(500))

    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(500))

    description: Mapped[str] = mapped_column(Text)

    location: Mapped[str | None] = mapped_column(String(500))
    work_format: Mapped[str | None] = mapped_column(String(50))
    employment_type: Mapped[str | None] = mapped_column(String(50))
    seniority: Mapped[str | None] = mapped_column(String(50))

    salary_from: Mapped[int | None]
    salary_to: Mapped[int | None]
    salary_currency: Mapped[str | None] = mapped_column(String(10))

    skills: Mapped[list[dict]] = mapped_column(
        JSON,
        default=list,
    )

    requirements: Mapped[list[dict]] = mapped_column(
        JSON,
        default=list,
    )

    responsibilities: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
    )

    benefits: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
    )

    raw_data: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
    )

    published_at: Mapped[datetime | None]
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
