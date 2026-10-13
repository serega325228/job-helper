from dataclasses import dataclass
from uuid import UUID

from pydantic import Field
from pydantic.main import BaseModel

from src.schemas.resume import ResumeData
from src.services.profile import ProfileService
from src.services.resume import ResumeService

# Append all questions for creating completed resume
QUESTIONS = [
    ("name", "What's your name?"),
    ("title", "What's the title of position you're applying for?"),
]

class MasterResumeState(BaseModel):
    messages: list[str] = Field(default_factory=list)
    language: str
    profile_id: UUID = Field()
    create_resume_manually: str | None = Field(default=None)
    resume: ResumeData # Maybe add default
    current_section: str
    next_question: str
    is_complete: bool
    last_answer: str



@dataclass(slots=True)
class MasterResumeContext:
    resume_service: ResumeService
    profile_service: ProfileService
