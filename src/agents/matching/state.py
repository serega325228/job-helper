from dataclasses import dataclass
from uuid import UUID

from pydantic.main import BaseModel
from services.profile import ProfileService
from services.scoring import ProfileComparison, ScoringService
from services.vacancy import VacancyService


class MatchingState(BaseModel):
    profile_id: UUID
    preference_intent_ids: list[UUID]
    hard_filters:
    limit: int

    profile_comparison: ProfileComparison

    vacancy_match_ids: list[UUID]


@dataclass
class MatchingContext:
    profile_service: ProfileService
    vacancy_service: VacancyService
    scoring_service: ScoringService
