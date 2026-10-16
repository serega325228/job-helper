from dataclasses import dataclass
from operator import add
from uuid import UUID

from langgraph.graph.message import Annotated, add_messages
from pydantic import Field
from pydantic.main import BaseModel

from src.schemas.resume import ResumeData
from src.services.profile import ProfileService
from src.services.resume import ResumeService


class MasterResumeState(BaseModel):
    messages: Annotated[list[str], add_messages] = Field(default_factory=list)
    raw_answers: Annotated[list[dict], add] = Field(default_factory=list)
    language: str
    profile_id: UUID = Field()
    create_resume_manually: str | None = Field(default=None)
    resume: ResumeData  # Maybe add default
    missing_fields: list[str] = Field(default_factory=list)
    current_section: str
    next_question: str
    is_complete: bool
    last_answer: str


@dataclass(slots=True)
class MasterResumeContext:
    resume_service: ResumeService
    profile_service: ProfileService
