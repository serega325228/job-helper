from dataclasses import dataclass
from uuid import UUID

from pydantic.main import BaseModel

from schemas.scoring import PreferenceComparison, ProfileComparison
from schemas.vacancy import VacancyHardFilters
from schemas.vacancy_match import VacancyEmbeddingSearchResult
from services.profile import ProfileService
from services.scoring import ScoringService
from services.vacancy import VacancyService
from services.vacancy_match import VacancyMatchService


class MatchingState(BaseModel):
    profile_id: UUID
    hard_filters: VacancyHardFilters
    limit: int

    vacancy_ids_after_hard_filters: list[UUID]
    vacancy_embedding_search_result: dict[UUID, VacancyEmbeddingSearchResult]
    compared_vacancies: dict[UUID, tuple[ProfileComparison, PreferenceComparison]]

    vacancy_match_ids: list[UUID]


@dataclass
class MatchingContext:
    profile_service: ProfileService
    vacancy_service: VacancyService
    scoring_service: ScoringService
    matching_service: VacancyMatchService
