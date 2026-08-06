from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, HttpUrl, model_validator


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


class VacancySearchQuery(BaseModel):
    text: str = Field(min_length=1)
    area_ids: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    published_after: datetime | None = None


class VacancyReference(BaseModel):
    source: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    published_at: datetime | None = None


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
    skills: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    additional_conditions: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    completeness_score: float = Field(default=0.0, ge=0, le=1)


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
