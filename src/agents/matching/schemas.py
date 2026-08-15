from uuid import UUID

from pydantic import BaseModel, Field

from src.schemas.scoring import PreferenceComparison, ProfileComparison


class MatchingCandidate(BaseModel):
    preference_id: UUID
    title_similarity: float
    content_similarity: float
    embedding_similarity: float

    profile_comparison: ProfileComparison | None = None
    preference_comparison: PreferenceComparison | None = None
    structured_score: float | None = Field(default=None, ge=0, le=1)

    profile_rerank_score: float | None = Field(default=None, ge=0, le=1)
    preference_rerank_score: float | None = Field(default=None, ge=0, le=1)
