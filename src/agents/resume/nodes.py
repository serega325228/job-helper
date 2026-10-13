from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.agents.resume.state import QUESTIONS, MasterResumeContext, MasterResumeState

async def choose_creation_method(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    answer = interrupt("Do you want to create a master resume manually or based on your profile?")
    return {"create_resume_manually": answer}

async def prepare_question(state: MasterResumeState, runtime: Runtime[MasterResumeContext]) -> dict:
    out = await runtime.context.resume_service.prepare_resume_question(
        state.language,
        state.current_section,
        state.last_answer,
        state.resume,
    )

    return {
        "resume": out.resume_data,
        "current_section": out.next_question.section,
        "next_question": out.next_question.text,
        "is_complete": out.is_complete,
    }

async def ask_user(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    answer = interrupt(state.next_question)
    return {"last_answer": answer}

async def save_resume(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    resume = await runtime.context.resume_service.master_resume_from_data(
        profile_id=state.profile_id,
        suffix=state.suffix,
        content=state.content,
        title=state.title,
        language=state.language,
        raw_data=state.raw_data,
        data=state.data,
    )

    await runtime.context.resume_service.save_resume(resume)

    return {"messages": [("ai", "Resume saved successfully.")]}


async def create_manually(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:


async def analyze_profile(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
