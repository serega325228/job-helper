from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import TypeAdapter

from src.agents.profile.state import ProfileContext, ProfileState
from src.schemas.preference import PreferenceData
from src.schemas.profile import ProfileData


async def start(state: ProfileState, runtime: Runtime[ProfileContext]) -> dict:
    profile = await runtime.context.profile_service.get_profile_data(state.profile_id)
    return {"profile": profile, "approved": False}


async def prepare_question(
    state: ProfileState, runtime: Runtime[ProfileContext]
) -> dict:
    out = await runtime.context.profile_service.analyze_data(
        profile_id=state.profile_id,
        language=state.language,
        data=state.profile,
        last_answer=state.last_answer,
        last_question=state.next_question,
        raw_answers=state.raw_answers,
        preference_text=state.preference_text,
    )
    drafts = out.preferences
    if drafts:
        drafts = await runtime.context.profile_service.prepare_preference_drafts(state.profile_id, drafts)
    return {
        "profile": out.profile_data,
        "preferences": drafts,
        "preferences_approved": False,
        "next_question": out.next_question,
        "missing_fields": out.missing_fields,
        "approved": False,
    }


async def ask_user(state: ProfileState, runtime: Runtime[ProfileContext]) -> dict:
    question = state.next_question
    if not state.missing_fields or question is None:
        raise ValueError("Cannot ask a question without unresolved missing_fields")
    payload = {"kind": "question", **question.model_dump()}
    while True:
        answer = interrupt(payload)
        if isinstance(answer, str):
            break
        payload = {"kind": "question", **question.model_dump(), "error": "Please answer with text."}
    # Only the next analysis can decide whether this answer resolves the gap.
    return {
        "last_answer": answer,
        "raw_answers": [{**question.model_dump(), "answer": answer}],
    }


async def profile_review(state: ProfileState, runtime: Runtime[ProfileContext]) -> dict:
    payload = {
        "kind": "profile_review",
        "profile": state.profile.model_dump(mode="json"),
        "preferences": [item.model_dump(mode="json") for item in state.preferences],
        "message": "Please review and edit if needed. Edits will be re-analyzed before approval.",
    }
    while True:
        decision = interrupt(payload)
        try:
            if not isinstance(decision, dict):
                raise ValueError("The review must be an object")
            approved = decision.get("approved", False)
            if not isinstance(approved, bool):
                raise ValueError("approved must be a boolean")
            profile = ProfileData.model_validate(decision.get("profile", state.profile))
            preferences = TypeAdapter(list[PreferenceData]).validate_python(
                decision.get("preferences", state.preferences)
            )
            break
        except ValueError:
            payload = {
                **payload,
                "error": "Provide valid profile/preferences and a boolean approved value.",
            }
    # The original story is immutable even if the reviewed extraction changes.
    profile = profile.model_copy(update={"raw_data": state.profile.raw_data})
    if profile != state.profile:
        answer = (
            "The user corrected the extracted profile to these values, "
            "including any cleared fields: " + profile.model_dump_json()
        )
        return {
            "profile": profile,
            "preferences": preferences,
            "approved": False,
            "last_answer": answer,
            "raw_answers": [{"field": "profile", "text": "Review profile", "answer": answer}],
        }
    return {"profile": profile, "preferences": preferences, "approved": approved}


async def preferences_review(state: ProfileState, runtime: Runtime[ProfileContext]) -> dict:
    existing = await runtime.context.profile_service.get_preferences_by_profile_id(state.profile_id)
    payload = {
        "kind": "preferences_review",
        "drafts": [item.model_dump(mode="json") for item in state.preferences],
        "existing": [
            {"id": str(item.id), "enabled": item.enabled, **PreferenceData.model_validate(item).model_dump(mode="json")}
            for item in existing
        ],
        "message": "Approve or edit new preferences, or decline to keep existing preferences only.",
    }
    while True:
        decision = interrupt(payload)
        try:
            if not isinstance(decision, dict) or not isinstance(decision.get("approved"), bool):
                raise ValueError("A boolean approved value is required")
            drafts = TypeAdapter(list[PreferenceData]).validate_python(
                decision.get("preferences", state.preferences)
            ) if decision["approved"] else []
            break
        except ValueError:
            payload = {**payload, "error": "Provide valid preferences and a boolean approved value."}
    return {"preferences": drafts, "preferences_approved": decision["approved"]}


async def save_profile(state: ProfileState, runtime: Runtime[ProfileContext]) -> dict:
    if state.missing_fields or not state.approved:
        raise ValueError("Only a reviewed profile with no missing_fields can be saved")
    if state.preferences and not state.preferences_approved:
        raise ValueError("New preferences require separate confirmation")
    await runtime.context.profile_service.update_profile(
        profile_id=state.profile_id,
        data=state.profile,
        preferences=state.preferences,
        expected_revision=state.source_revision,
        save_key=state.save_key,
    )
    return {"saved": True, "messages": [("ai", "Profile saved successfully.")]}
