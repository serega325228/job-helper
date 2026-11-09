from dataclasses import dataclass
from operator import add
from typing import Annotated
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agents.schemas import NextQuestion
from src.schemas.preference import PreferenceData
from src.schemas.profile import ProfileData
from src.services.profile import ProfileService


class ProfileState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    raw_answers: Annotated[list[dict], add] = Field(default_factory=list)
    language: str
    profile_id: UUID
    profile: ProfileData = Field(
        default_factory=lambda: ProfileData(
            name="", raw_data="", processed_data={}, profile_summary=""
        )
    )
    preferences: list[PreferenceData] = Field(default_factory=list)
    # Initial [] is not used for routing until the analysis node has run.
    missing_fields: list[str] = Field(default_factory=list)
    next_question: NextQuestion | None = None
    last_answer: str = ""
    approved: bool = False


@dataclass(slots=True)
class ProfileContext:
    profile_service: ProfileService
