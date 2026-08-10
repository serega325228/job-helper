from pydantic import BaseModel, Field


class PreferenceIntentAnalysis(BaseModel):
    name: str
    description: str | None = None
    target_titles: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    preferred_industries: list[str] = Field(default_factory=list)
    preferred_companies: list[str] = Field(default_factory=list)
    excluded_titles: list[str] = Field(default_factory=list)
    excluded_companies: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    work_formats: list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)
    salary_min: int | None = Field(default=None, ge=0)
    salary_currency: str | None = None
    min_seniority: str | None = None
    max_seniority: str | None = None


class ProfileAnalysis(BaseModel):
    summary: str
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    seniority: str | None = None
    experience_years: float | None = Field(default=None, ge=0)
    preference_intents: list[PreferenceIntentAnalysis] = Field(
        default_factory=list,
    )
    missing_information: list[str] = Field(default_factory=list)
