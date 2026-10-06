from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Enum, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, String, Text

from src.infrastructure.models.base import Base
from src.schemas.vacancy import CollectionStatus, PreviewStatus


class PreviewCollection(Base):
    __tablename__ = "preview_collections"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("profiles.id"), index=True)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("companies.id"))
    source: Mapped[str] = mapped_column(String(50))
    vacancy_page_url: Mapped[str] = mapped_column(Text)
    preference_ids: Mapped[list[str]] = mapped_column(JSONB)
    query: Mapped[dict[str, Any]] = mapped_column(JSONB)
    hard_filters: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[CollectionStatus] = mapped_column(
        Enum(
            CollectionStatus,
            native_enum=False,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        default=CollectionStatus.PENDING,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (Index("ix_preview_collections_recovery", "status", "updated_at"),)


class VacancyPreview(Base):
    __tablename__ = "vacancy_previews"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    collection_id: Mapped[UUID] = mapped_column(
        ForeignKey("preview_collections.id"), index=True
    )
    source: Mapped[str] = mapped_column(String(50))
    external_id: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(500))
    company_name: Mapped[str | None] = mapped_column(String(500))
    normalized_company: Mapped[str | None] = mapped_column(Text)
    normalized_title: Mapped[str] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    short_description: Mapped[str | None] = mapped_column(Text)
    work_format: Mapped[str | None] = mapped_column(String(50))
    employment_type: Mapped[str | None] = mapped_column(String(50))
    experience: Mapped[str | None] = mapped_column(String(100))
    seniority: Mapped[str | None] = mapped_column(String(50))
    salary: Mapped[str | None] = mapped_column(Text)
    salary_from: Mapped[int | None]
    salary_to: Mapped[int | None]
    salary_currency: Mapped[str | None] = mapped_column(String(10))
    salary_gross: Mapped[bool | None]
    status: Mapped[PreviewStatus] = mapped_column(
        Enum(
            PreviewStatus,
            native_enum=False,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        index=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        UniqueConstraint(
            "source", "external_id", name="uq_vacancy_previews_source_external_id"
        ),
        Index(
            "ix_vacancy_previews_recent_reposts",
            "normalized_company",
            "normalized_title",
            "discovered_at",
        ),
        Index("ix_vacancy_previews_recovery", "status", "updated_at"),
    )
