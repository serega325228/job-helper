from dataclasses import dataclass
from uuid import UUID

from pydantic.main import BaseModel

from schemas.scoring import PreferenceComparison, ProfileComparison
from schemas.vacancy import VacancyHardFilters
from services.profile import ProfileService
from services.scoring import ScoringService
from services.vacancy import VacancyService


class MatchingState(BaseModel):
    profile_id: UUID
    hard_filters: VacancyHardFilters
    limit: int

    vacancy_ids_after_hard_filters: list[UUID]
    preference_id: UUID
    compared_vacancies: dict[UUID, tuple[ProfileComparison, PreferenceComparison]]

    vacancy_match_ids: list[UUID]


@dataclass
class MatchingContext:
    profile_service: ProfileService
    vacancy_service: VacancyService
    scoring_service: ScoringService
