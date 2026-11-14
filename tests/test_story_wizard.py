import json
import unittest
from typing import cast
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from dishka import Provider, Scope, make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from test_master_resume import Guard, Session, Store

from src.agents.profile.graph import create_profile_agent
from src.agents.profile.schemas import ProfileOutput
from src.agents.resume.graph import create_resume_agent
from src.agents.resume.schemas import ResumeOutput
from src.agents.schemas import NextQuestion
from src.exceptions.config import LLMError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.document_parser.document_parser import DocumentParser
from src.infrastructure.llm.llm import LLMProvider
from src.infrastructure.models.preference import Preference
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.resume import Resume
from src.infrastructure.resume_turn_guard import ResumeTurnGuard
from src.repositories.profile import ProfileRepository
from src.repositories.prompt import PromptRepository
from src.repositories.resume import ResumeRepository
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_match import VacancyMatchRepository
from src.repositories.vacancy_preview import VacancyPreviewRepository
from src.routers.resume import router as resume_router
from src.routers.wizard import router
from src.schemas.preference import PreferenceData
from src.schemas.profile import ProfileData
from src.schemas.resume import PersonalInfo, ResumeData
from src.services.master_resume import MasterResumeService
from src.services.preference_catalog import canonical_title, reconcile_preferences
from src.services.profile import ProfileService
from src.services.resume import ResumeService
from src.services.wizard import WizardService

STORY = "  I am Ada. I build Python services. Original story.\n"


def preference(**updates):
    return PreferenceData(general_title="Backend Developer", description="Explicit search goal", **updates)


def analysis(name="Ada", missing=False, preferences=None):
    return ProfileOutput(
        profile_data=ProfileData(
            name=name, raw_data="LLM must not replace source", processed_data={},
            profile_summary="Builds Python services", skills=["Python"],
        ),
        preferences=preferences or [],
        missing_fields=["name"] if missing else [],
        next_question=NextQuestion(field="name", text="What is your name?") if missing else None,
    )


def resume_analysis(missing=False):
    return ResumeOutput(
        resume_data=ResumeData(personalInfo=PersonalInfo(name="Ada", title="Engineer")),
        missing_fields=["summary"] if missing else [],
        next_question=NextQuestion(field="summary", text="Describe an achievement?") if missing else None,
    )


class WizardSession(Session):
    async def commit(self):
        for row in self.pending:
            if isinstance(row, Profile):
                self.store.profiles[row.id] = row
            elif isinstance(row, Resume):
                self.store.resumes[row.id] = row
        for profile in self.store.profiles.values():
            for item in profile.preferences:
                if item.id is None:
                    item.id = uuid4()
                if item.enabled is None:
                    item.enabled = True
        self.pending.clear()
        self.store.lock.release()

    async def scalars(self, stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        params = stmt.compile().params
        self.store.statements.append(sql)
        rows = self.store.profiles[params["profile_id_1"]].preferences
        return list(rows)


class CatalogTests(unittest.TestCase):
    def test_aliases_custom_titles_constraints_and_collision(self):
        self.assertEqual(canonical_title("  backend  developer "), "Backend Engineer")
        self.assertEqual(canonical_title("Senior Python Engineer"), "Senior Python Engineer")
        existing = preference(target_titles=["Backend Engineer"], work_formats=["remote"], locations=["Berlin"]).model_copy(update={"general_title": "Backend Engineer"})
        same = preference(target_titles=["Backend Developer"], work_formats=["REMOTE"], locations=[" Berlin "])
        self.assertEqual(reconcile_preferences([same], [existing]), [])
        distinct = same.model_copy(update={"locations": ["Paris"]})
        drafts = reconcile_preferences([distinct, distinct], [existing])
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].locations, ["Paris"])
        self.assertNotEqual(drafts[0].general_title, canonical_title(existing.general_title))
        no_targets = preference().model_copy(update={"general_title": "Backend Engineer"})
        different = no_targets.model_copy(update={"locations": ["Paris"]})
        additions = reconcile_preferences([different], [no_targets])
        self.assertEqual(reconcile_preferences([different], [no_targets, *additions]), [])
        different_prose = no_targets.model_copy(update={"description": "Exclude banks"})
        self.assertEqual(len(reconcile_preferences([different_prose], [no_targets])), 1)


class StoryWizardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store()
        self.checkpointer = InMemorySaver()
        self.profile_graph = create_profile_agent(self.checkpointer)
        self.resume_graph = create_resume_agent(self.checkpointer)
        self.guard = cast(ResumeTurnGuard, Guard())
        self.llm = Mock(spec=LLMProvider, complete=AsyncMock())
        self.prompts = Mock(spec=PromptRepository, get=Mock(side_effect=lambda key, default: (default, False)))
        self.contexts = []
        dependencies = Provider()
        dependencies.provide(self.wizard, provides=WizardService, scope=Scope.REQUEST)
        dependencies.provide(self.master, provides=MasterResumeService, scope=Scope.REQUEST)
        dependencies.provide(self.resumes, provides=ResumeService, scope=Scope.REQUEST)
        self.container = make_async_container(dependencies)
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        self.app.include_router(resume_router, prefix="/api/v1")
        setup_dishka(self.container, self.app)
        self.client = AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.container.close()

    def uow(self):
        session = cast(AsyncSession, WizardSession(self.store))
        return SqlAlchemyUnitOfWork(
            session, ProfileRepository(session), Mock(spec=VacancyRepository),
            Mock(spec=VacancyMatchRepository), ResumeRepository(session),
            Mock(spec=VacancyPreviewRepository),
        )

    def profiles(self):
        return ProfileService(self.uow(), self.prompts, self.llm)

    def wizard(self) -> WizardService:
        service = WizardService(self.profile_graph, self.guard, self.profiles())
        self.contexts.append(service._context)
        return service

    def resumes(self) -> ResumeService:
        return ResumeService(self.uow(), Mock(spec=DocumentParser), self.prompts, self.llm)

    def master(self) -> MasterResumeService:
        return MasterResumeService(self.resume_graph, self.guard, self.resumes(), self.profiles())

    @staticmethod
    def base(payload):
        return f"/api/v1/wizard/{payload['profile_id']}/threads/{payload['thread_id']}"

    async def start(self, **updates):
        response = await self.client.post("/api/v1/wizard/threads", json={"story": STORY, **updates})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    async def answer(self, payload, answer):
        return await self.client.post(self.base(payload) + "/answers", json={"answer": answer})

    async def approve(self, payload):
        response = await self.answer(payload, {"kind": "profile_review", "approved": True})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_story_followup_review_preferences_master_and_read(self):
        draft = preference(target_titles=["Backend Developer"], work_formats=["remote"])
        self.llm.complete.side_effect = [analysis("", True), analysis(preferences=[draft]), resume_analysis(True), resume_analysis()]
        payload = await self.start(preference_text="I want remote backend work")
        self.assertEqual(payload["interrupt"]["kind"], "question")
        profile_id = UUID(payload["profile_id"])
        denied = await self.client.post(self.base(payload) + "/master")
        self.assertEqual(denied.status_code, 409)
        response = await self.answer(payload, "Ada")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["interrupt"]["kind"], "profile_review")
        self.assertEqual(self.store.profiles[profile_id].processed_data, None)
        payload = await self.approve(payload)
        self.assertEqual(payload["interrupt"]["kind"], "preferences_review")
        self.assertEqual(payload["interrupt"]["drafts"][0]["general_title"], "Backend Engineer")
        self.assertEqual(self.store.profiles[profile_id].preferences, [])
        response = await self.answer(payload, {"kind": "preferences_review", "approved": True})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["actions"], ["create_master_resume"])
        stored = self.store.profiles[profile_id]
        self.assertEqual(stored.raw_data, STORY)
        self.assertEqual(stored.skills, ["Python"])
        self.assertEqual(len(stored.preferences), 1)
        self.assertEqual(len(self.store.resumes), 0)
        calls = self.llm.complete.await_args_list
        self.assertEqual(json.loads(calls[1].args[0])["answer_history"][0]["answer"], "Ada")
        self.assertEqual(json.loads(calls[1].args[0])["preference_text"], "I want remote backend work")
        handoff = await self.client.post(self.base(payload) + "/master")
        self.assertEqual(handoff.status_code, 200, handoff.text)
        self.assertEqual(handoff.json()["interrupt"]["field"], "summary")
        thread_id = handoff.json()["thread_id"]
        completed = await self.client.post(
            f"/api/v1/resume/{profile_id}/master/threads/{thread_id}/answers",
            json={"answer": "Built reliable services"},
        )
        self.assertEqual(completed.status_code, 200, completed.text)
        self.assertEqual(completed.json()["status"], "completed")
        master = completed.json()["master"]
        self.assertEqual((await self.client.get(f"/api/v1/resume/{profile_id}/master")).json(), master)
        self.assertEqual((await self.client.get(self.base(payload))).json()["actions"], [])
        self.assertEqual((await self.client.post(self.base(payload) + "/master")).status_code, 409)
        self.assertEqual(stored.raw_data, STORY)
        self.assertIsNot(self.contexts[0], self.contexts[1])
        self.assertTrue(any("FOR UPDATE" in sql for sql in self.store.statements))

    async def test_no_preferences_and_sufficient_profile_create_master_immediately(self):
        self.llm.complete.side_effect = [analysis(), resume_analysis()]
        payload = await self.approve(await self.start())
        self.assertEqual(payload["status"], "completed")
        result = await self.client.post(self.base(payload) + "/master")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "completed")
        self.assertEqual(len(self.store.resumes), 1)
        self.assertEqual(len(self.store.profiles[UUID(payload["profile_id"])].preferences), 0)

    async def test_existing_preferences_reused_distinct_appended_and_rejection_preserves(self):
        existing_data = preference(target_titles=["Backend Engineer"], locations=["Berlin"])
        existing = Preference(id=uuid4(), profile_id=self.store.profile_id, enabled=True, **existing_data.model_dump())
        self.store.profiles[self.store.profile_id].preferences = [existing]
        for approve in (False, True):
            new = preference(target_titles=["Backend Developer"], locations=["Paris"])
            self.llm.complete.return_value = analysis(preferences=[existing_data, new])
            started = await self.client.post(f"/api/v1/wizard/{self.store.profile_id}/threads", json={"language": "ru"})
            self.assertEqual(started.status_code, 201, started.text)
            payload = await self.approve(started.json())
            self.assertEqual(len(payload["interrupt"]["drafts"]), 1)
            self.assertEqual(len(payload["interrupt"]["existing"]), 1)
            response = await self.answer(payload, {"kind": "preferences_review", "approved": approve})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(self.store.profiles[self.store.profile_id].preferences), 1 + approve)
            self.assertIs(self.store.profiles[self.store.profile_id].preferences[0], existing)
            self.assertEqual(existing.locations, ["Berlin"])
            if approve:
                new_title = self.store.profiles[self.store.profile_id].preferences[1].general_title
                self.assertNotEqual(new_title, existing.general_title)
                retry = await self.answer(payload, {"kind": "preferences_review", "approved": True})
                self.assertEqual(retry.status_code, 200, retry.text)
                self.assertEqual(len(self.store.profiles[self.store.profile_id].preferences), 2)

    async def test_edits_reanalyzed_and_invalid_or_foreign_answers_rejected(self):
        self.llm.complete.side_effect = [analysis(), analysis("", True), analysis("Grace")]
        payload = await self.start()
        wrong = await self.answer(payload, "approve")
        self.assertEqual(wrong.status_code, 422)
        edited = payload["profile"]
        edited["name"] = ""
        edited["raw_data"] = "must not overwrite"
        response = await self.answer(payload, {"kind": "profile_review", "approved": True, "profile": edited})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["interrupt"]["kind"], "question")
        self.assertNotIn("_wizard_save_key", response.json()["profile"])
        response = await self.answer(payload, "Grace")
        payload = await self.approve(response.json())
        self.assertEqual(payload["profile"]["raw_data"], STORY)
        foreign = await self.client.get(f"/api/v1/wizard/{self.store.profile_id}/threads/{payload['thread_id']}")
        self.assertEqual(foreign.status_code, 404)
        unknown = await self.client.get(self.base({**payload, "thread_id": str(uuid4())}))
        self.assertEqual(unknown.status_code, 404)
        schema_reject = await self.client.post(self.base(payload) + "/answers", json={"answer": "text", "language": "ru"})
        self.assertEqual(schema_reject.status_code, 422)

    async def test_initial_failure_and_failed_answer_are_recoverable_after_recompile(self):
        self.llm.complete.side_effect = [LLMError("provider_unavailable"), analysis("", True), LLMError("provider_unavailable"), analysis()]
        failed = await self.client.post("/api/v1/wizard/threads", json={"story": STORY})
        self.assertEqual(failed.status_code, 503, failed.text)
        payload = {"profile_id": failed.headers["X-Wizard-Profile-ID"], "thread_id": failed.headers["X-Wizard-Thread-ID"]}
        self.profile_graph = create_profile_agent(self.checkpointer)
        recovered = await self.answer(payload, "not a new answer")
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["interrupt"]["kind"], "question")
        failed = await self.answer(payload, "Ada")
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("Ada", failed.text)
        self.assertEqual((await self.client.get(self.base(payload))).json()["interrupt"]["kind"], "retry")
        recovered = await self.answer(payload, "must not be recorded")
        self.assertEqual(recovered.status_code, 200, recovered.text)
        request = json.loads(self.llm.complete.await_args.args[0])
        self.assertEqual([item["answer"] for item in request["answer_history"]], ["Ada"])
        self.assertEqual(len(self.store.profiles), 2)

    async def test_post_commit_failure_retries_without_duplicate_preferences(self):
        self.llm.complete.return_value = analysis(preferences=[preference()])
        payload = await self.approve(await self.start())
        original = ProfileService.update_profile

        async def crash_after_commit(service, *args, **kwargs):
            await original(service, *args, **kwargs)
            raise RuntimeError("private crash after commit")

        with patch.object(ProfileService, "update_profile", crash_after_commit):
            failed = await self.answer(payload, {"kind": "preferences_review", "approved": True})
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("private", failed.text)
        stored = self.store.profiles[UUID(payload["profile_id"])]
        self.assertEqual(len(stored.preferences), 1)
        recovered = await self.answer(payload, "retry pending save")
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["status"], "completed")
        self.assertEqual(len(stored.preferences), 1)
        self.assertEqual(self.llm.complete.await_count, 1)

    async def test_stale_review_cannot_overwrite_newer_profile(self):
        self.llm.complete.return_value = analysis()
        first = await self.start()
        profile_id = first["profile_id"]
        other = await self.client.post(f"/api/v1/wizard/{profile_id}/threads", json={})
        self.assertEqual(other.status_code, 201, other.text)
        await self.approve(first)
        stale = await self.answer(other.json(), {"kind": "profile_review", "approved": True})
        self.assertEqual(stale.status_code, 409, stale.text)
        self.assertEqual(stale.json()["detail"], "wizard_profile_changed")

    async def test_equivalent_enabled_preferences_reused_without_new_review(self):
        data = preference(target_titles=["Backend Engineer"])
        existing = Preference(id=uuid4(), profile_id=self.store.profile_id, enabled=True, **data.model_dump())
        self.store.profiles[self.store.profile_id].preferences = [existing]
        self.llm.complete.return_value = analysis(preferences=[data])
        started = await self.client.post(f"/api/v1/wizard/{self.store.profile_id}/threads", json={})
        self.assertEqual(started.status_code, 201, started.text)
        payload = await self.approve(started.json())
        self.assertEqual(payload["status"], "completed")
        self.assertEqual([item["id"] for item in payload["preferences"]], [str(existing.id)])
        self.assertEqual(self.store.profiles[self.store.profile_id].preferences, [existing])

    async def test_disabled_preference_not_reused_or_implicitly_enabled(self):
        data = preference()
        existing = Preference(id=uuid4(), profile_id=self.store.profile_id, enabled=False, **data.model_dump())
        self.store.profiles[self.store.profile_id].preferences = [existing]
        self.llm.complete.return_value = analysis(preferences=[data])
        started = await self.client.post(f"/api/v1/wizard/{self.store.profile_id}/threads", json={})
        payload = await self.approve(started.json())
        self.assertEqual(payload["interrupt"]["kind"], "preferences_review")
        response = await self.answer(payload, {"kind": "preferences_review", "approved": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(existing.enabled)
        self.assertEqual(len(self.store.profiles[self.store.profile_id].preferences), 2)

    async def test_preference_edits_are_validated_and_preserved(self):
        self.llm.complete.return_value = analysis(preferences=[preference()])
        payload = await self.approve(await self.start())
        draft = payload["interrupt"]["drafts"][0]
        invalid = {**draft, "salary_min": -1}
        response = await self.answer(payload, {"kind": "preferences_review", "approved": True, "preferences": [invalid]})
        self.assertEqual(response.status_code, 422)
        response = await self.answer(payload, {"kind": "preferences_review", "approved": True, "preferences": [{**draft, "unknown": "not stored"}]})
        self.assertEqual(response.status_code, 422)
        draft["locations"] = ["Paris"]
        response = await self.answer(payload, {"kind": "preferences_review", "approved": True, "preferences": [draft]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["preferences"][0]["locations"], ["Paris"])

    async def test_master_handoff_failure_exposes_resume_thread_for_retry(self):
        self.llm.complete.side_effect = [analysis(), LLMError("provider_unavailable"), resume_analysis()]
        payload = await self.approve(await self.start())
        failed = await self.client.post(self.base(payload) + "/master")
        self.assertEqual(failed.status_code, 503, failed.text)
        thread_id = failed.headers["X-Resume-Thread-ID"]
        recovered = await self.client.post(
            f"/api/v1/resume/{payload['profile_id']}/master/threads/{thread_id}/answers",
            json={"answer": "retry"},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["status"], "completed")

    async def test_catalog_and_schema_validation(self):
        from src.di.container import create_container
        from src.di.providers import AgentProvider
        from src.main import app

        container = create_container()
        await container.close()
        paths = app.openapi()["paths"]
        for path in (
            "/api/v1/wizard/preference-titles",
            "/api/v1/wizard/threads",
            "/api/v1/wizard/{profile_id}/threads",
            "/api/v1/wizard/{profile_id}/threads/{thread_id}",
            "/api/v1/wizard/{profile_id}/threads/{thread_id}/answers",
            "/api/v1/wizard/{profile_id}/threads/{thread_id}/master",
        ):
            self.assertIn(path, paths)
        provider = AgentProvider()
        self.assertIs(provider.profile_agent(self.checkpointer).checkpointer, self.checkpointer)
        self.assertIs(provider.resume_agent(self.checkpointer).checkpointer, self.checkpointer)
        catalog = await self.client.get("/api/v1/wizard/preference-titles")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json()["version"], "1")
        self.assertIn("backend-engineer", [item["id"] for item in catalog.json()["roles"]])
        for body in ({"story": " "}, {"story": "x", "language": "zz"}, {"story": "x", "profile_id": str(uuid4())}):
            response = await self.client.post("/api/v1/wizard/threads", json=body)
            self.assertEqual(response.status_code, 422)
        self.llm.complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
