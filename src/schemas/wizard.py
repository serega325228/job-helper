from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from src.infrastructure.models.resume import ResumeLanguage
from src.schemas.preference import PreferenceData, RoleTitle
from src.schemas.profile import ProfileData


class StartWizardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story: str = Field(min_length=1, max_length=100_000)
    name: str = Field(default="", max_length=255)
    language: ResumeLanguage = ResumeLanguage.EN
    preference_text: str = Field(default="", max_length=100_000)

    @field_validator("story")
    @classmethod
    def nonblank_story(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Story must not be blank")
        return value  # Preserve the original source verbatim.


class StartProfileWizardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: ResumeLanguage = ResumeLanguage.EN
    preference_text: str = Field(default="", max_length=100_000)


class ProfileReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["profile_review"]
    approved: bool = Field(strict=True)
    profile: ProfileData | None = None


class ReviewedPreferenceData(PreferenceData):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class SavedPreference(PreferenceData):
    id: UUID
    enabled: bool


class PreferencesReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["preferences_review"]
    approved: bool = Field(strict=True)
    preferences: list[ReviewedPreferenceData] | None = None


class AnswerWizardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | Annotated[
        ProfileReviewDecision | PreferencesReviewDecision, Field(discriminator="kind")
    ]

    @field_validator("answer")
    @classmethod
    def valid_text(
        cls, value: str | ProfileReviewDecision | PreferencesReviewDecision
    ) -> str | ProfileReviewDecision | PreferencesReviewDecision:
        if isinstance(value, str) and (not value.strip() or len(value) > 100_000):
            raise ValueError("Answer must contain between 1 and 100000 nonblank characters")
        return value


class WizardTurnResponse(BaseModel):
    profile_id: UUID
    thread_id: UUID
    language: str
    status: Literal["interrupted", "completed"]
    interrupt: dict[str, JsonValue] | None = None
    profile: ProfileData
    missing_fields: list[str] = Field(default_factory=list)
    preferences: list[SavedPreference] = Field(default_factory=list)
    master_resume_available: bool = False
    actions: list[Literal["create_master_resume"]] = Field(default_factory=list)


class PreferenceTitleCatalogResponse(BaseModel):
    version: str
    roles: tuple[RoleTitle, ...]
