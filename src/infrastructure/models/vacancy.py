from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Float, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, String, Text

from src.infrastructure.models.base import Base


class Vacancy(Base):
    __tablename__ = "vacancies"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    company_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("companies.id"),
        nullable=True,
    )

    source: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    company_name: Mapped[str | None] = mapped_column(String(500), index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Hard conditions stay queryable without inspecting JSON documents.
    area_id: Mapped[str | None] = mapped_column(String(100), index=True)
    country: Mapped[str | None] = mapped_column(String(255), index=True)
    city: Mapped[str | None] = mapped_column(String(255), index=True)
    work_format: Mapped[str | None] = mapped_column(String(50), index=True)
    employment_type: Mapped[str | None] = mapped_column(String(50), index=True)
    work_schedule: Mapped[str | None] = mapped_column(String(100), index=True)
    experience: Mapped[str | None] = mapped_column(String(100), index=True)
    seniority: Mapped[str | None] = mapped_column(String(50), index=True)

    salary_from: Mapped[int | None] = mapped_column(index=True)
    salary_to: Mapped[int | None] = mapped_column(index=True)
    salary_currency: Mapped[str | None] = mapped_column(String(10), index=True)
    salary_gross: Mapped[bool | None] = mapped_column(Boolean)

    soft_conditions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    raw_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    completeness_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    normalizer_version: Mapped[str | None] = mapped_column(String(100))

    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
        index=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    last_seen_at: Mapped[datetime] = mapped_column(
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

    __table_args__ = (
        UniqueConstraint(
            "source",
            "external_id",
            name="uq_vacancies_source_external_id",
        ),
        Index(
            "ix_vacancies_soft_conditions_gin",
            "soft_conditions",
            postgresql_using="gin",
        ),
    )
