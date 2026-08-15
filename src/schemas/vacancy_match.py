from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class MatchCategory(StrEnum):
    TARGET = "target"
    STRETCH = "stretch"
    FALLBACK = "fallback"
    REJECT = "reject"


class VacancyEmbeddingSearchResult(BaseModel):
    vacancy_id: UUID
    preference_id: UUID
    title_similarity: float
    content_similarity: float
    combined_similarity: float


class VacancyMatchResult(BaseModel):
    profile_id: UUID
    vacancy_id: UUID
    preference_intent_id: UUID
    structured_profile_score: float = Field(ge=0, le=1)
    structured_preference_score: float = Field(ge=0, le=1)
    profile_rerank_score: float | None = Field(default=None, ge=0, le=1)
    preference_rerank_score: float | None = Field(default=None, ge=0, le=1)
    total_score: float = Field(ge=0, le=1)
    category: MatchCategory
    hard_constraints_passed: bool = True
    component_scores: dict[str, Any] = Field(default_factory=dict)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str | None = None
    matcher_version: str = "structured-rerank-v1"
    reranker_model: str | None = None
