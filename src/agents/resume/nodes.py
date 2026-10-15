import json

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.agents.resume.state import MasterResumeContext, MasterResumeState


async def choose_creation_method(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    answer = interrupt(
        "Do you want to create a master resume manually or based on your profile?"
    )
    return {"create_resume_manually": answer}


async def manually_prepare_question(
    state: MasterResumeState, runtime: Runtime[MasterResumeContext]
) -> dict:
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
        "missing_fields": out.missing_fields,
    }


async def based_on_profile_prepare_question(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    out = await runtime.context.profile_service.analyze_profile_and_prepare_resume_question(
        state.profile_id,
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
        "missing_fields": out.missing_fields,
    }


async def ask_user(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    field = state.missing_fields[0]
    answer = interrupt(field)
    return {
        "last_answer": answer,
        "missing_fields": state.missing_fields[1:],
        "raw_answers": [{"field": field, "answer": answer}],
    }


async def save_resume(
    state: MasterResumeState,
    runtime: Runtime[MasterResumeContext],
) -> dict:
    resume = runtime.context.resume_service.master_resume_from_data(
        profile_id=state.profile_id,
        title=state.resume.personal_info.title,
        language=state.language,
        raw_data=json.dumps(state.raw_answers),
        data=state.resume,
    )

    await runtime.context.resume_service.save_resume(resume)

    return {"messages": [("ai", "Resume saved successfully.")]}
