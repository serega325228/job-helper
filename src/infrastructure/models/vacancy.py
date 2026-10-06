from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Enum, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Boolean, DateTime, String, Text

from src.infrastructure.models.base import Base
from src.ports.embedder import EMBEDDING_DIMENSIONS
from src.schemas.vacancy import BatchStatus, ProcessingStatus, VacancyStatus

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
    normalized_company: Mapped[str | None] = mapped_column(
        Text,
        Computed(
            "lower(trim(regexp_replace(company_name, '\\s+', ' ', 'g')))",
            persisted=True,
        ),
    )
    normalized_title: Mapped[str] = mapped_column(
        Text,
        Computed(
            "lower(trim(regexp_replace(title, '\\s+', ' ', 'g')))", persisted=True
        ),
    )
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

    status: Mapped[VacancyStatus] = mapped_column(
        Enum(
            VacancyStatus,
            native_enum=False,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=VacancyStatus.ACTIVE,
        nullable=False,
        index=True,
    )
    preview_id: Mapped[UUID | None] = mapped_column(ForeignKey("vacancy_previews.id"))
    batch_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("vacancy_batches.id"), index=True
    )
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(
            ProcessingStatus,
            native_enum=False,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=ProcessingStatus.COMPLETED,
        index=True,
    )
    raw_document: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    evaluation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
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
        Index(
            "ix_vacancies_recent_reposts",
            "normalized_company",
            "normalized_title",
            "discovered_at",
        ),
        Index("ix_vacancies_recovery", "processing_status", "updated_at"),
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


class VacancyBatch(Base):
    __tablename__ = "vacancy_batches"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("profiles.id"))
    preference_ids: Mapped[list[str]] = mapped_column(JSONB)
    hard_filters: Mapped[dict[str, Any]] = mapped_column(JSONB)
    search_limit: Mapped[int]
    rerank_limit: Mapped[int]
    title_weight: Mapped[float]
    status: Mapped[BatchStatus] = mapped_column(
        Enum(
            BatchStatus,
            native_enum=False,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=BatchStatus.DETAILS,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (Index("ix_vacancy_batches_recovery", "status", "updated_at"),)
