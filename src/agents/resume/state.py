from dataclasses import dataclass
from operator import add
from typing import Annotated, Literal
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agents.schemas import NextQuestion
from src.schemas.resume import PersonalInfo, ResumeData
from src.services.profile import ProfileService
from src.services.resume import ResumeService


class MasterResumeState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    raw_answers: Annotated[list[dict], add] = Field(default_factory=list)
    language: str
    profile_id: UUID
    creation_method: Literal["manual", "profile"] | None = None
    resume: ResumeData = Field(
        default_factory=lambda: ResumeData(personalInfo=PersonalInfo())
    )
    # Initial [] is not used for routing until the analysis node has run.
    missing_fields: list[str] = Field(default_factory=list)
    next_question: NextQuestion | None = None
    last_answer: str = ""
    save_key: UUID | None = None
    resume_id: UUID | None = None


@dataclass(slots=True)
class MasterResumeContext:
    resume_service: ResumeService
    profile_service: ProfileService
