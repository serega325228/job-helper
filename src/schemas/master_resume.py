from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.infrastructure.models.resume import Resume, ResumeLanguage
from src.schemas.resume import ResumeData


class StartMasterResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: ResumeLanguage
    creation_method: Literal["manual", "profile"] | None = None


class AnswerMasterResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=100_000)


class MasterResumeResponse(BaseModel):
    id: UUID
    profile_id: UUID
    language: str
    title: str | None
    filename: str | None
    raw_data: str
    data: ResumeData


class ResumeInterrupt(BaseModel):
    field: str | None = None
    text: str | None = None
    question: str | None = None
    options: list[str] | None = None
    error: str | None = None


class MasterResumeTurnResponse(BaseModel):
    thread_id: UUID
    profile_id: UUID
    language: str
    status: Literal["interrupted", "completed"]
    interrupt: ResumeInterrupt | None = None
    master: MasterResumeResponse | None = None


def master_resume_response(resume: Resume) -> MasterResumeResponse:
    return MasterResumeResponse(
        id=resume.id,
        profile_id=resume.profile_id,
        language=resume.language,
        title=resume.title,
        filename=resume.filename,
        raw_data=resume.raw_data,
        data=ResumeData.model_validate(resume.processed_data),
    )
