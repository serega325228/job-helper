from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import JsonValue

from src.agents.profile.state import ProfileContext, ProfileState
from src.exceptions.profile import (
    ProfileNotFoundError,
    ProfileTurnError,
    ProfileWorkflowError,
)
from src.infrastructure.resume_turn_guard import ResumeTurnGuard
from src.schemas.profile import ProfileData
from src.schemas.wizard import (
    AnswerWizardRequest,
    SavedPreference,
    StartProfileWizardRequest,
    StartWizardRequest,
    WizardTurnResponse,
)
from src.services.profile import ProfileService, profile_revision


class WizardService:
    """Typed profile graph transport; no request services are checkpointed."""

    def __init__(
        self,
        graph: CompiledStateGraph[ProfileState, ProfileContext, ProfileState, ProfileState],
        guard: ResumeTurnGuard,
        profile_service: ProfileService,
    ) -> None:
        self._graph = graph
        self._guard = guard
        self._profiles = profile_service
        self._context = ProfileContext(profile_service)

    @staticmethod
    def _config(thread_id: UUID) -> RunnableConfig:
        return {"configurable": {"thread_id": f"profile-wizard:{thread_id}"}}

    async def start(self, request: StartWizardRequest) -> WizardTurnResponse:
        profile = await self._profiles.create(request.name, request.story)
        return await self.start_profile(profile.id, StartProfileWizardRequest(
            language=request.language, preference_text=request.preference_text,
        ))

    async def start_profile(
        self, profile_id: UUID, request: StartProfileWizardRequest
    ) -> WizardTurnResponse:
        thread_id = uuid4()
        config = self._config(thread_id)
        try:
            async with self._guard.hold(profile_id):
                profile = await self._profiles.get_profile(profile_id)
                if profile is None:
                    raise ProfileNotFoundError(profile_id)
                await self._graph.ainvoke(
                    ProfileState(
                        profile_id=profile_id,
                        language=request.language.value,
                        preference_text=request.preference_text,
                        source_revision=profile_revision(profile),
                        save_key=str(uuid4()),
                    ),
                    config=config, context=self._context,
                )
                return await self._response(profile_id, thread_id, config)
        except Exception as error:
            raise ProfileTurnError(profile_id, thread_id, error) from error

    async def _state(self, profile_id: UUID, thread_id: UUID) -> StateSnapshot:
        if await self._profiles.get_profile(profile_id) is None:
            raise ProfileNotFoundError(profile_id)
        state = await self._graph.aget_state(self._config(thread_id))
        if not state.values or str(state.values.get("profile_id")) != str(profile_id):
            raise ProfileWorkflowError("wizard_thread_not_found", 404)
        return state

    async def get(self, profile_id: UUID, thread_id: UUID) -> WizardTurnResponse:
        await self._state(profile_id, thread_id)
        return await self._response(profile_id, thread_id, self._config(thread_id))

    async def answer(
        self, profile_id: UUID, thread_id: UUID, request: AnswerWizardRequest
    ) -> WizardTurnResponse:
        async with self._guard.hold(profile_id):
            state = await self._state(profile_id, thread_id)
            config = self._config(thread_id)
            if not state.next and state.values.get("saved"):
                return await self._response(profile_id, thread_id, config)
            if not state.interrupts and not state.next:
                raise ProfileWorkflowError("wizard_thread_not_waiting", 409)
            answer = request.answer
            if state.interrupts:
                kind = state.interrupts[0].value.get("kind", "question")
                if isinstance(answer, str):
                    valid = kind == "question"
                else:
                    valid = answer.kind == kind
                if not valid:
                    raise ProfileWorkflowError("wizard_answer_type_mismatch", 422)
            # A failed node already has the last answer durably checkpointed.
            # Retry it with None rather than appending a duplicate correction.
            value = answer if isinstance(answer, str) else answer.model_dump(
                mode="json", exclude_none=True
            )
            await self._graph.ainvoke(
                Command(resume=value) if state.interrupts else None,
                config=config, context=self._context,
            )
            return await self._response(profile_id, thread_id, config)

    async def ensure_master_available(self, profile_id: UUID, thread_id: UUID) -> str:
        async with self._guard.hold(profile_id):
            state = await self._state(profile_id, thread_id)
            if state.next or not state.values.get("saved") or state.values.get("missing_fields"):
                raise ProfileWorkflowError("wizard_profile_not_ready", 409)
            if await self._profiles.has_master_resume(profile_id):
                raise ProfileWorkflowError("master_resume_exists", 409)
            return state.values["language"]

    async def _response(
        self, profile_id: UUID, thread_id: UUID, config: RunnableConfig
    ) -> WizardTurnResponse:
        state = await self._graph.aget_state(config)
        completed = not state.next and bool(state.values.get("saved"))
        available = completed and not await self._profiles.has_master_resume(profile_id)
        payload: dict[str, JsonValue] | None = state.interrupts[0].value if state.interrupts else None
        # GET on a failed, pending turn is recoverable via the answer endpoint.
        if not completed and payload is None:
            payload = {"kind": "retry", "message": "Retry the pending turn with the answer endpoint."}
        return WizardTurnResponse(
            profile_id=profile_id, thread_id=thread_id,
            language=state.values["language"],
            status="completed" if completed else "interrupted",
            interrupt=payload,
            profile=ProfileData.model_validate(state.values["profile"]),
            missing_fields=state.values.get("missing_fields", []),
            preferences=[SavedPreference.model_validate(item) for item in
                         await self._profiles.get_preferences_by_profile_id(profile_id)],
            master_resume_available=available,
            actions=["create_master_resume"] if available else [],
        )
