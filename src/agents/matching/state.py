from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, Field

from src.agents.matching.schemas import MatchingCandidate
from src.schemas.vacancy import VacancyHardFilters
from src.services.profile import ProfileService
from src.services.scoring import ScoringService
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService


class MatchingState(BaseModel):
    profile_id: UUID
    hard_filters: VacancyHardFilters = Field(default_factory=VacancyHardFilters)
    search_limit: int = Field(default=100, ge=1)
    ann_candidate_limit: int = Field(default=100, ge=1)
    per_preference_limit: int = Field(default=50, ge=1)
    rerank_limit: int = Field(default=40, ge=1)
    title_weight: float = Field(default=0.4, ge=0, le=1)

    candidates: dict[UUID, MatchingCandidate] = Field(default_factory=dict)
    vacancy_match_ids: list[UUID] = Field(default_factory=list)


@dataclass(slots=True)
class MatchingContext:
    profile_service: ProfileService
    vacancy_service: VacancyService
    scoring_service: ScoringService
    matching_service: VacancyMatchService
