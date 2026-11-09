from src.agents.schemas import FollowUpOutput
from src.schemas.resume import ResumeData


class ResumeOutput(FollowUpOutput):
    resume_data: ResumeData
