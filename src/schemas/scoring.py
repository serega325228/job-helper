from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProfileComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)

class LayaComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    role_fit: Literal["none", "weak", "good", "strong"]
    skill_fit: Literal["none", "weak", "good", "strong"]

class PreferenceComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=1)
    hard_constraints_passed: bool
    components: dict[str, float] = Field(default_factory=dict)


class VacancyRerankScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_score: float = Field(ge=0, le=1)
    preference_score: float = Field(ge=0, le=1)
