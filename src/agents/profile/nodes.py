import json

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.agents.profile.state import ProfileContext, ProfileState


async def start(state: ProfileState, runtime: Runtime[ProfileContext]) -> dict:
    profile = await runtime.context.profile_service.get_profile(state.profile_id)
    return {
        "profile_exists": profile is not None,
    }


async def prepare_question(
    state: ProfileState, runtime: Runtime[ProfileContext]
) -> dict:
    out = await runtime.context.profile_service.prepare_question(
        state.language,
        state.current_section,
        state.last_answer,
        state.profile,
    )

    return {
        "resume": out.resume_data,
        "current_section": out.next_question.section,
        "next_question": out.next_question.text,
        "is_complete": out.is_complete,
        "missing_fields": out.missing_fields,
    }


async def ask_user(
    state: ProfileState,
    runtime: Runtime[ProfileContext],
) -> dict:
    field = state.missing_fields[0]
    answer = interrupt(field)
    return {
        "last_answer": answer,
        "missing_fields": state.missing_fields[1:],
        "raw_answers": [{"field": field, "answer": answer}],
    }


async def profile_review(
    state: ProfileState,
    runtime: Runtime[ProfileContext],
) -> dict:
    decision = interrupt(
        {
            "profile": state.profile,
            "preferences": state.preferences,  # Move to separate node with separate llm invocation
            "message": "Please review and edit if needed.",
        }
    )
    return {"profile": decision.get("profile", state.profile), "approved": True}


# Save/update Profile from ProfileData, think about statuses here and in resume agent
async def save_profile(
    state: ProfileState,
    runtime: Runtime[ProfileContext],
) -> dict:
    resume = runtime.context.profile_service.master_resume_from_data(
        profile_id=state.profile_id,
        title=state.resume.personal_info.title,
        language=state.language,
        raw_data=json.dumps(state.raw_answers),
        data=state.resume,
    )

    await runtime.context.resume_service.save_resume(resume)

    return {"messages": [("ai", "Resume saved successfully.")]}
