from pydantic import BaseModel
from pydantic.config import ConfigDict
from pydantic.fields import Field


# Maybe rework/add specific field types
class PreferenceData(BaseModel):
    model_config = ConfigDict(extra="allow")

    general_title: str
    description: str
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
    salary_min: int | None = Field(default=None)
    salary_currency: str | None = Field(default=None)
    min_seniority: str | None = Field(default=None)
    max_seniority: str | None = Field(default=None)
