from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel


class MatchCategory(StrEnum):
    TARGET = "target"
    STRETCH = "stretch"
    FALLBACK = "fallback"
    REJECT = "reject"


class VacancyMatchResult(BaseModel):
    vacancy_id: UUID

    profile_fit: float
    preference_fit: float
    semantic_fit: float | None = None

    final_score: float

    matched_intent_id: UUID
    category: MatchCategory

    missing_required_skills: list[str]
    matched_skills: list[str]
