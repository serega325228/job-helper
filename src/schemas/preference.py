from pydantic import BaseModel, field_validator
from pydantic.config import ConfigDict
from pydantic.fields import Field


class RoleTitle(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    aliases: tuple[str, ...] = ()


# Maybe rework/add specific field types
class PreferenceData(BaseModel):
    model_config = ConfigDict(extra="allow", from_attributes=True)

    general_title: str = Field(min_length=1, max_length=200)
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
    salary_min: int | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None)
    min_seniority: str | None = Field(default=None)
    max_seniority: str | None = Field(default=None)

    @field_validator("general_title")
    @classmethod
    def nonblank_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Preference title must not be blank")
        return value.strip()
