from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.agents.resume.state import MasterResumeContext, MasterResumeState
from src.exceptions.resume import ResumeError, ResumeTurnError
from src.infrastructure.resume_turn_guard import ResumeTurnGuard
from src.schemas.master_resume import (
    MasterResumeTurnResponse,
    ResumeInterrupt,
    StartMasterResumeRequest,
    master_resume_response,
)
from src.services.profile import ProfileService
from src.services.resume import ResumeService


class MasterResumeService:
    """Direct graph transport. Request services are passed anew on every invocation."""

    def __init__(
        self,
        graph: CompiledStateGraph[
            MasterResumeState, MasterResumeContext, MasterResumeState, MasterResumeState
        ],
        guard: ResumeTurnGuard,
        resume_service: ResumeService,
        profile_service: ProfileService,
    ) -> None:
        self._graph = graph
        self._guard = guard
        self._resumes = resume_service
        self._context = MasterResumeContext(resume_service, profile_service)

    @staticmethod
    def _config(thread_id: UUID) -> RunnableConfig:
        return {"configurable": {"thread_id": f"master-resume:{thread_id}"}}

    async def start(
        self, profile_id: UUID, request: StartMasterResumeRequest
    ) -> MasterResumeTurnResponse:
        async with self._guard.hold(profile_id):
            await self._resumes.ensure_profile(profile_id, creating=True)
            thread_id = uuid4()
            config = self._config(thread_id)
            try:
                await self._graph.ainvoke(
                    MasterResumeState(
                        profile_id=profile_id,
                        language=request.language.value,
                        creation_method=request.creation_method,
                        save_key=uuid4(),
                    ),
                    config=config,
                    context=self._context,
                )
                return await self._response(profile_id, thread_id, config)
            except Exception as error:
                # Even the first analysis/save can fail after a checkpoint or commit.
                # Retain its server-issued handle so the caller can retry this thread.
                raise ResumeTurnError(thread_id, error) from error

    async def answer(
        self, profile_id: UUID, thread_id: UUID, answer: str
    ) -> MasterResumeTurnResponse:
        async with self._guard.hold(profile_id):
            await self._resumes.ensure_profile(profile_id)
            config = self._config(thread_id)
            state = await self._graph.aget_state(config)
            # Unknown and foreign threads are deliberately indistinguishable.
            if not state.values or str(state.values.get("profile_id")) != str(profile_id):
                raise ResumeError("resume_thread_not_found", 404)
            if not state.next and state.values.get("resume_id"):
                return await self._response(profile_id, thread_id, config)
            save_key = state.values.get("save_key")
            await self._resumes.ensure_profile(
                profile_id, creating=True,
                resume_id=UUID(str(save_key)) if save_key else None,
            )
            if not state.interrupts and not state.next:
                raise ResumeError("resume_thread_not_waiting", 409)
            # On a failed analysis/save the answer is already checkpointed. Retry the
            # pending node without appending the submitted text a second time.
            await self._graph.ainvoke(
                Command(resume=answer) if state.interrupts else None,
                config=config, context=self._context,
            )
            return await self._response(profile_id, thread_id, config)

    async def _response(
        self, profile_id: UUID, thread_id: UUID, config: RunnableConfig
    ) -> MasterResumeTurnResponse:
        state = await self._graph.aget_state(config)
        if state.interrupts:
            return MasterResumeTurnResponse(
                thread_id=thread_id,
                profile_id=profile_id,
                language=state.values["language"],
                status="interrupted",
                interrupt=ResumeInterrupt.model_validate(state.interrupts[0].value),
            )
        resume_id = state.values.get("resume_id")
        if state.next or not resume_id:
            raise ResumeError("resume_turn_incomplete", 503)
        master = await self._resumes.get_master(profile_id, UUID(str(resume_id)))
        return MasterResumeTurnResponse(
            thread_id=thread_id,
            profile_id=profile_id,
            language=state.values["language"],
            status="completed",
            master=master_resume_response(master),
        )
