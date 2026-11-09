from pydantic import Field

from src.agents.schemas import FollowUpOutput
from src.schemas.preference import PreferenceData
from src.schemas.profile import ProfileData


class ProfileOutput(FollowUpOutput):
    profile_data: ProfileData
    preferences: list[PreferenceData] = Field(default_factory=list)
