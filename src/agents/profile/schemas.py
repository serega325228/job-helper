from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.preference import PreferenceData
from src.schemas.profile import ProfileData

Section = Literal[
    "name",
    "raw_data",
    "processed_data",
    "profile_summary",
    "skills",
    "experience",
    "education",
    "seniority",
    "experience_years",
    "contacts",
    "custom_sections",
    "section_meta",
]

class NextQuestion(BaseModel):
    text: str
    section: Section

class ProfileOutput(BaseModel):
    profile_data: ProfileData
    preferences: list[PreferenceData] = Field(default_factory=list) # Move to separate node with separate llm invocation
    next_question: NextQuestion
    inferred_skills: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    is_complete: bool = False
