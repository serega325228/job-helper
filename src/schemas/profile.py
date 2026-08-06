from typing import Any

from pydantic import BaseModel, Field


class ProfileAnalysis(BaseModel):
    summary: str
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    target_titles: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
