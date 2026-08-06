from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class VacancySearchQuery(BaseModel):
    text: str
    area_ids: list[str] = []
    experience: list[str] = []
    published_after: datetime | None = None

class VacancyReference(BaseModel):
    source: str
    external_id: str
    title: str
    url: HttpUrl
    published_at: datetime | None = None

class RawVacancy(BaseModel):
    source: str
    external_id: str
    url: HttpUrl

    title: str | None = None
    raw_text: str | None = None
    raw_payload: dict[str, Any] | None = None

    fetched_at: datetime

class NormalizedVacancy(BaseModel):
    source: str
    external_id: str

    title: str
    company_name: str | None = None
    description: str

    skills: list[str] = []
    responsibilities: list[str] = []
    requirements: list[str] = []

    salary_from: int | None = None
    salary_to: int | None = None
    salary_currency: str | None = None

    location: str | None = None
    remote: bool | None = None
    experience_level: str | None = None

    extraction_warnings: list[str] = []
    completeness_score: float = Field(ge=0, le=1)

class NormalizedVacancyBatch(BaseModel):
    items: list[NormalizedVacancy]
