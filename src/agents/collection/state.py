from dataclasses import dataclass

from infrastructure.models.vacancy import Vacancy
from ports.vacancy_source import VacancySource
from pydantic.main import BaseModel
from schemas.vacancy import NormalizedVacancy, RawVacancy, VacancySearchQuery
from services.vacancy import VacancyService


class CollectionState(BaseModel):
    sources: list[VacancySource]
    queries: list[VacancySearchQuery]
    raw_vacancies: list[RawVacancy]
    normalized_vacancies: list[NormalizedVacancy]
    saved_vacancies: list[Vacancy]

@dataclass
class CollectionContext:
    vacancy_service: VacancyService
