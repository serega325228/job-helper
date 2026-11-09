from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

Strategy = Literal["keywords", "full", "cheat"]


class PersonalInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    title: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    website: str | None = None
    linkedin: str | None = None
    github: str | None = None


class WorkExperience(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | int | None = None
    title: str | None = None
    company: str | None = None
    location: str | None = None
    years: str | None = None
    description: list[str] = Field(default_factory=list)
    descriptionStyles: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("descriptionStyles", "description_styles"),
    )


class Education(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | int | None = None
    institution: str | None = None
    degree: str | None = None
    years: str | None = None
    description: str | None = None


class PersonalProject(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | int | None = None
    name: str | None = None
    role: str | None = None
    years: str | None = None
    description: list[str] = Field(default_factory=list)
    descriptionStyles: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("descriptionStyles", "description_styles"),
    )


class AdditionalInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    technicalSkills: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("technicalSkills", "technical_skills"),
    )
    languages: list[str] = Field(default_factory=list)
    certificationsTraining: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("certificationsTraining", "certifications_training"),
    )
    awards: list[str] = Field(default_factory=list)



# TODO: add description to fields for llm
class ResumeData(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Accept old stored keys, but expose and serialize only the canonical camelCase fields.
    personalInfo: PersonalInfo = Field(
        validation_alias=AliasChoices("personalInfo", "personal_info")
    )
    summary: str | None = None
    workExperience: list[WorkExperience] = Field(
        default_factory=list,
        validation_alias=AliasChoices("workExperience", "work_experience"),
    )
    education: list[Education] = Field(default_factory=list)
    personalProjects: list[PersonalProject] = Field(
        default_factory=list,
        validation_alias=AliasChoices("personalProjects", "personal_projects"),
    )
    additional: AdditionalInfo = Field(default_factory=AdditionalInfo)
    customSections: dict[str, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("customSections", "custom_sections"),
    )
    sectionMeta: dict[str, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("sectionMeta", "section_meta"),
    )


class JobKeywords(BaseModel):
    company: str = ""
    role: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    key_responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0)
    seniority_level: str | None = None


class ResumeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    action: Literal["replace", "append", "reorder", "add_skill"]
    original: str | None
    value: str | list[str]
    reason: str


class ImproveDiffResult(BaseModel):
    changes: list[ResumeChange]
    strategy_notes: str = ""


class ProtectedClaimsReview(BaseModel):
    rejected_indices: list[int] = Field(
        description="Zero-based indices of edits inventing or changing protected personal facts."
    )


class RejectedChange(BaseModel):
    change: ResumeChange
    reason: str


class SkillTarget(BaseModel):
    skill: str
    source: Literal["existing", "supported_by_resume", "unverified"]


class ImprovementSuggestion(BaseModel):
    suggestion: str
    status: Literal["applied", "recommended"]
    field_path: str | None = None
    unverified: bool = False


class ImprovementResult(BaseModel):
    resume: ResumeData
    applied_changes: list[ResumeChange] = Field(default_factory=list)
    rejected_changes: list[RejectedChange] = Field(default_factory=list)
    suggestions: list[ImprovementSuggestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unverified_skills: list[str] = Field(default_factory=list)
    strategy_notes: str = ""
