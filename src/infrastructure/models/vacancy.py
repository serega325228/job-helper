from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Boolean, DateTime, String, Text

from src.infrastructure.models.base import Base
from src.infrastructure.models.constants import EMBEDDING_DIMENSIONS

if TYPE_CHECKING:
    from src.infrastructure.models.vacancy_match import VacancyMatch


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
    content_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
    )
    title_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
    )

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

    vacancy_matches: Mapped[list[VacancyMatch]] = relationship(
        back_populates="vacancy",
        cascade="all, delete-orphan",
        passive_deletes=True,
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
        Index(
            "ix_vacancies_content_embedding_hnsw",
            "content_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"content_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_vacancies_title_embedding_hnsw",
            "title_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"title_embedding": "vector_cosine_ops"},
        ),
    )
