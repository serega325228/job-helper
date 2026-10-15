from pydantic import BaseModel, ConfigDict, Field


# Create profile data schema
class ProfileData(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    raw_story: str = Field(Text)
    profile_summary: Mapped[str | None] = mapped_column(Text)
    analysis_status: Mapped[str] = mapped_column(  # Maybe delete this
        String(30),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
    )

    skills: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    experience: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    education: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    seniority: Mapped[str | None] = mapped_column(String(50), index=True)
    experience_years: Mapped[float | None] = mapped_column(Float)
    contacts: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )


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
