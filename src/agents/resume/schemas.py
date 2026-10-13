from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.resume import ResumeData

Section = Literal[
    "intro", "contact", "summary", "workExperience",
    "internships", "education", "personalProjects",
    "skills", "review",
]

class NextQuestion(BaseModel):
    text: str
    section: Section

class WizardTurnOutput(BaseModel):
    resume_data: ResumeData
    next_question: NextQuestion
    inferred_skills: list[str] = Field(default_factory=list)
    is_complete: bool = False
