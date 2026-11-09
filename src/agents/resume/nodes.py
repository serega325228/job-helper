import json

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.agents.resume.state import MasterResumeContext, MasterResumeState


async def choose_creation_method(
    state: MasterResumeState, runtime: Runtime[MasterResumeContext]
) -> dict:
    if state.creation_method is not None:
        return {}
    payload = {
        "question": "Create a master resume manually or based on your profile?",
        "options": ["manual", "profile"],
    }
    while True:
        answer = interrupt(payload)
        method = answer.strip().lower() if isinstance(answer, str) else None
        if method in ("manual", "profile"):
            return {"creation_method": method}
        payload = {**payload, "error": "Choose 'manual' or 'profile'."}


async def manually_prepare_question(
    state: MasterResumeState, runtime: Runtime[MasterResumeContext]
) -> dict:
    out = await runtime.context.resume_service.prepare_resume_question(
        language=state.language,
        last_question=state.next_question,
        last_answer=state.last_answer,
        data=state.resume,
        raw_answers=state.raw_answers,
    )
    return {
        "resume": out.resume_data,
        "next_question": out.next_question,
        "missing_fields": out.missing_fields,
    }


async def based_on_profile_prepare_question(
    state: MasterResumeState, runtime: Runtime[MasterResumeContext]
) -> dict:
    out = await runtime.context.profile_service.analyze_profile_and_prepare_resume_question(
        profile_id=state.profile_id,
        language=state.language,
        last_question=state.next_question,
        last_answer=state.last_answer,
        resume_data=state.resume,
        raw_answers=state.raw_answers,
    )
    return {
        "resume": out.resume_data,
        "next_question": out.next_question,
        "missing_fields": out.missing_fields,
    }


async def ask_user(state: MasterResumeState, runtime: Runtime[MasterResumeContext]) -> dict:
    question = state.next_question
    if not state.missing_fields or question is None:
        raise ValueError("Cannot ask a question without unresolved missing_fields")
    payload = question.model_dump()
    while True:
        answer = interrupt(payload)
        if isinstance(answer, str):
            break
        payload = {**question.model_dump(), "error": "Please answer with text."}
    return {
        "last_answer": answer,
        "raw_answers": [{**question.model_dump(), "answer": answer}],
    }


async def save_resume(
    state: MasterResumeState, runtime: Runtime[MasterResumeContext]
) -> dict:
    if state.missing_fields:
        raise ValueError("Cannot save a resume with unresolved missing_fields")
    resume = runtime.context.resume_service.master_resume_from_data(
        profile_id=state.profile_id,
        title=state.resume.personalInfo.title or "",
        language=state.language,
        raw_data=json.dumps(state.raw_answers, ensure_ascii=False),
        data=state.resume,
    )
    await runtime.context.resume_service.save_resume(resume)
    return {
        "resume_id": resume.id,
        "messages": [("ai", "Resume saved successfully.")],
    }
