from pydantic import BaseModel, Field


class InterviewPrepQuestion(BaseModel):
    """Interview question grounded in the tailored resume and job context."""

    question: str
    focus_area: str | None = None
    suggested_answer_points: list[str] = Field(default_factory=list)


class InterviewPrepSkillGap(BaseModel):
    """A preparation target, not a claimed candidate skill."""

    skill: str
    why_it_matters: str
    preparation_suggestion: str


class InterviewPrepData(BaseModel):
    """Structured interview preparation content for a tailored resume."""

    role_fit_analysis: list[str]
    resume_questions: list[InterviewPrepQuestion]
    project_follow_ups: list[InterviewPrepQuestion]
    skill_gaps: list[InterviewPrepSkillGap]
    talking_points: list[str]
