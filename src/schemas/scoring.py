from pydantic import BaseModel, ConfigDict, Field


class ProfileComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)


class PreferenceComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=1)
    hard_constraints_passed: bool
    components: dict[str, float] = Field(default_factory=dict)


class VacancyRerankScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_score: float = Field(ge=0, le=1)
    preference_score: float = Field(ge=0, le=1)
