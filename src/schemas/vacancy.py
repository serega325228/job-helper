from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class WorkFormat(StrEnum):
    ON_SITE = "on_site"
    REMOTE = "remote"
    HYBRID = "hybrid"
    OTHER = "other"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    OTHER = "other"


class PreviewStatus(StrEnum):
    PENDING_FILTER = "pending_filter"
    READY = "ready"
    REJECTED = "rejected"
    SELECTED = "selected"


class VacancyStatus(StrEnum):
    ACTIVE = "active"
    REMOVED = "removed"
    ARCHIVED = "archived"


class ProcessingStatus(StrEnum):
    PENDING_SCRAPE = "pending_scrape"
    PENDING_PARSE = "pending_parse"
    PENDING_EMBEDDING = "pending_embedding"
    PENDING_RERANK = "pending_rerank"
    PENDING_LAYA = "pending_laya"
    PENDING_SAVE = "pending_save"
    COMPLETED = "completed"
    FILTERED = "filtered"
    REMOVED = "removed"


class CollectionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


class BatchStatus(StrEnum):
    DETAILS = "details"
    PENDING_RERANK = "pending_rerank"
    PENDING_LAYA = "pending_laya"
    PENDING_SAVE = "pending_save"
    COMPLETED = "completed"


class VacancySearchQuery(BaseModel):
    text: str = Field(min_length=1)
    area_ids: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    published_after: datetime | None = None


class VacancyScrapingQuery(VacancySearchQuery):
    model_config = ConfigDict(str_strip_whitespace=True)

    location: str | None = Field(default=None, min_length=1, max_length=255)
    required_keywords: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )
    excluded_keywords: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )


class VacancyHardFilters(BaseModel):
    sources: list[str] = Field(default_factory=list)
    statuses: list[VacancyStatus] = Field(
        default_factory=lambda: [VacancyStatus.ACTIVE]
    )
    area_ids: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    company_names: list[str] = Field(default_factory=list)
    excluded_company_names: list[str] = Field(default_factory=list)

    work_formats: list[WorkFormat] = Field(default_factory=list)
    employment_types: list[EmploymentType] = Field(default_factory=list)
    work_schedules: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    seniorities: list[str] = Field(default_factory=list)

    salary_min: int | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=1, max_length=10)
    salary_gross: bool | None = None
    published_after: datetime | None = None


class VacancyReference(BaseModel):
    source: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    published_at: datetime | None = None


class VacancyPreview(VacancyReference):
    model_config = ConfigDict(from_attributes=True)
    company_name: str | None = Field(default=None, max_length=500)
    location: str | None = None
    short_description: str | None = None
    work_format: WorkFormat | None = None
    employment_type: EmploymentType | None = None
    experience: str | None = None
    seniority: str | None = None
    salary: str | None = None
    salary_from: int | None = Field(default=None, ge=0)
    salary_to: int | None = Field(default=None, ge=0)
    salary_currency: str | None = None
    salary_gross: bool | None = None


class PreviewCollectionRequest(BaseModel):
    profile_id: UUID
    sources: list[str] = Field(default_factory=lambda: ["all"], min_length=1)
    preferences: list[str] = Field(default_factory=lambda: ["all"], min_length=1)
    vacancy_page_urls: dict[str, HttpUrl] = Field(default_factory=dict)
    query: VacancyScrapingQuery
    hard_filters: VacancyHardFilters = Field(default_factory=VacancyHardFilters)


class PreviewSelectionRequest(BaseModel):
    profile_id: UUID
    preview_ids: list[UUID] | None = None
    hard_filters: VacancyHardFilters = Field(default_factory=VacancyHardFilters)
    limit: int = Field(default=50, ge=1, le=250)
    search_limit: int = Field(default=100, ge=1)
    rerank_limit: int = Field(default=40, ge=1)
    title_weight: float = Field(default=0.4, ge=0, le=1)


class PreviewResponse(VacancyPreview):
    id: UUID
    status: PreviewStatus


class VacancyScrapingResult(BaseModel):
    source: str
    vacancy_ids: list[UUID] = Field(default_factory=list)
    search_pages: int = 0
    previews_discovered: int = 0
    rejected_by_filters: int = 0
    detail_pages_enqueued: int = 0
    detail_pages_scraped: int = 0
    duplicates_skipped: int = 0
    failed_pages: int = 0
    removed_pages: int = 0
    vacancies_saved: int = 0


class RawVacancy(BaseModel):
    source: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    title: str | None = Field(default=None, max_length=500)
    raw_text: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    published_at: datetime | None = None
    fetched_at: datetime


class VacancySoftConditions(BaseModel):
    summary: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    additional_conditions: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)


class NormalizedVacancy(BaseModel):
    source: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    company_name: str | None = Field(default=None, max_length=500)
    description: str

    area_id: str | None = Field(default=None, max_length=100)
    country: str | None = Field(default=None, max_length=255)
    city: str | None = Field(default=None, max_length=255)
    work_format: WorkFormat | None = None
    employment_type: EmploymentType | None = None
    work_schedule: str | None = Field(default=None, max_length=100)
    experience: str | None = Field(default=None, max_length=100)
    seniority: str | None = Field(default=None, max_length=50)

    salary_from: int | None = Field(default=None, ge=0)
    salary_to: int | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, max_length=10)
    salary_gross: bool | None = None

    soft_conditions: VacancySoftConditions = Field(
        default_factory=VacancySoftConditions,
    )

    @model_validator(mode="after")
    def validate_salary_range(self) -> Self:
        if (
            self.salary_from is not None
            and self.salary_to is not None
            and self.salary_from > self.salary_to
        ):
            raise ValueError("salary_from must not exceed salary_to")
        return self


class NormalizedVacancyBatch(BaseModel):
    items: list[NormalizedVacancy] = Field(default_factory=list)
