from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Boolean, DateTime, String, Text

from src.infrastructure.models.base import Base
from src.ports.embedder import EMBEDDING_DIMENSIONS

if TYPE_CHECKING:
    from src.infrastructure.models.profile import Profile
    from src.infrastructure.models.vacancy_match import VacancyMatch


class Preference(Base):
    __tablename__ = "preferences"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    target_titles: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    required_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    preferred_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    preferred_industries: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    preferred_companies: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    excluded_titles: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    excluded_companies: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    locations: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    work_formats: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    employment_types: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    salary_min: Mapped[int | None]
    salary_currency: Mapped[str | None] = mapped_column(String(10))
    min_seniority: Mapped[str | None] = mapped_column(String(50))
    max_seniority: Mapped[str | None] = mapped_column(String(50))

    content_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
    )
    title_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
    )

    embedding_fingerprint: Mapped[str | None] = mapped_column(String(64))

    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
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

    profile: Mapped[Profile] = relationship(back_populates="preferences")
    vacancy_matches: Mapped[list[VacancyMatch]] = relationship(
        back_populates="preferences",
    )

    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "name",
            name="uq_preferences_profile_name",
        ),
        CheckConstraint("weight > 0", name="preferences_weight_positive"),
        Index(
            "ix_preferences_content_embedding_hnsw",
            "content_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"content_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_preferences_title_embedding_hnsw",
            "title_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"title_embedding": "vector_cosine_ops"},
        ),
    )
