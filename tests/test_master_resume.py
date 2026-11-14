import asyncio
import unittest
from contextlib import asynccontextmanager
from typing import cast
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from dishka import Provider, Scope, make_async_container
from dishka.integrations.fastapi import setup_dishka
from docling.exceptions import ConversionError
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import configure_mappers
from starlette.datastructures import UploadFile

from src.agents.resume.graph import create_resume_agent
from src.agents.resume.schemas import ResumeOutput
from src.agents.schemas import NextQuestion
from src.di.container import create_container
from src.di.providers import AgentProvider, checkpoint_conninfo
from src.exceptions.config import LLMError
from src.exceptions.resume import ResumeError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.document_parser.document_parser import DocumentParser
from src.infrastructure.llm.llm import LLMProvider
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.resume import Resume, ResumeType
from src.infrastructure.resume_turn_guard import ResumeTurnGuard
from src.repositories.profile import ProfileRepository
from src.repositories.prompt import PromptRepository
from src.repositories.resume import ResumeRepository
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_match import VacancyMatchRepository
from src.repositories.vacancy_preview import VacancyPreviewRepository
from src.routers.resume import MAX_UPLOAD_BYTES, router
from src.schemas.resume import PersonalInfo, ResumeData
from src.services.master_resume import MasterResumeService
from src.services.profile import ProfileService
from src.services.resume import ResumeService


def document(name="Ada"):
    return ResumeData(personalInfo=PersonalInfo(name=name, title="Engineer"))


def turn(missing=False):
    return ResumeOutput(
        resume_data=document(),
        missing_fields=["summary"] if missing else [],
        next_question=NextQuestion(field="summary", text="Describe your work?") if missing else None,
    )


class Store:
    def __init__(self):
        self.profile_id = uuid4()
        self.profiles = {
            self.profile_id: Profile(
                id=self.profile_id, name="Ada", raw_data="Original profile story",
                processed_data={}, skills=[], experience=[], education=[], contacts={},
            )
        }
        self.resumes = {}
        self.statements = []
        self.lock = asyncio.Lock()


class Session:
    """Executes the real repositories' bounded selects against an in-memory row set."""

    def __init__(self, store):
        self.store = store
        self.pending = []

    async def begin(self):
        await self.store.lock.acquire()

    async def commit(self):
        self.store.resumes.update({row.id: row for row in self.pending})
        self.pending.clear()
        self.store.lock.release()

    async def rollback(self):
        self.pending.clear()
        self.store.lock.release()

    def add(self, row):
        self.pending.append(row)

    async def flush(self):
        for row in self.pending:
            if row.id is None:
                row.id = uuid4()

    async def scalar(self, stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        params = stmt.compile().params
        self.store.statements.append(sql)
        if "FROM profiles" in sql:
            return self.store.profiles.get(params["id_1"])
        rows = list(self.store.resumes.values()) + self.pending
        rows = [row for row in rows if row.profile_id == params.get("profile_id_1")]
        if "type_1" in params:
            rows = [row for row in rows if row.type == params["type_1"]]
        if "id_1" in params:
            rows = [row for row in rows if row.id == params["id_1"]]
        if "SELECT EXISTS" in sql:
            return bool(rows)
        if "count(" in sql:
            return len(rows)
        return rows[0] if rows else None


class Guard:
    def __init__(self):
        self.locks = {}

    @asynccontextmanager
    async def hold(self, profile_id):
        async with self.locks.setdefault(profile_id, asyncio.Lock()):
            yield


class MasterResumeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store()
        self.checkpointer = InMemorySaver()
        self.graph = create_resume_agent(self.checkpointer)
        self.guard = cast(ResumeTurnGuard, Guard())
        self.llm = Mock(spec=LLMProvider, complete=AsyncMock())
        self.parser = Mock(spec=DocumentParser, parse=Mock(return_value="Original extracted resume"))
        self.prompts = Mock(spec=PromptRepository, get=Mock(side_effect=lambda key, default: (default, False)))
        self.contexts = []
        dependencies = Provider()
        dependencies.provide(self.resume_service, provides=ResumeService, scope=Scope.REQUEST)
        dependencies.provide(self.workflow, provides=MasterResumeService, scope=Scope.REQUEST)
        self.container = make_async_container(dependencies)
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        setup_dishka(self.container, self.app)
        self.client = AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test")
        self.base = f"/api/v1/resume/{self.store.profile_id}/master"

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.container.close()

    def uow(self):
        # The fake implements only the session operations exercised by these repositories.
        session = cast(AsyncSession, Session(self.store))
        return SqlAlchemyUnitOfWork(
            session, ProfileRepository(session),
            Mock(spec=VacancyRepository, get_amount=AsyncMock(return_value=0)),
            Mock(spec=VacancyMatchRepository, get_amount_by_profile_id=AsyncMock(return_value=0)),
            ResumeRepository(session), Mock(spec=VacancyPreviewRepository),
        )

    def resume_service(self) -> ResumeService:
        return ResumeService(self.uow(), self.parser, self.prompts, self.llm)

    def workflow(self) -> MasterResumeService:
        resumes = self.resume_service()
        profile = ProfileService(resumes._uow, self.prompts, self.llm)
        service = MasterResumeService(self.graph, self.guard, resumes, profile)
        self.contexts.append(service._context)
        return service

    async def start(self, method: str | None = "manual"):
        return await self.client.post(self.base + "/threads", json={
            "language": "en", "creation_method": method,
        })

    async def answer(self, thread_id, answer="Built reliable services"):
        return await self.client.post(self.base + f"/threads/{thread_id}/answers", json={"answer": answer})

    async def upload(self, filename="Original CV.DOCX", content=b"document", language="de"):
        return await self.client.post(self.base + "/import", data={"language": language}, files={
            "file": (filename, content, "application/octet-stream"),
        })

    async def test_manual_and_profile_create_answer_read_status(self):
        for method in ("manual", "profile"):
            with self.subTest(method=method):
                self.store.resumes.clear()
                self.llm.complete.side_effect = [turn(True), turn(False)]
                started = await self.start(method)
                self.assertEqual(started.status_code, 200, started.text)
                payload = started.json()
                self.assertEqual(payload["status"], "interrupted")
                self.assertEqual(payload["interrupt"]["field"], "summary")
                saved = await self.answer(payload["thread_id"])
                self.assertEqual(saved.status_code, 200, saved.text)
                result = saved.json()
                self.assertEqual(result["status"], "completed")
                master = result["master"]
                self.assertEqual(master["data"]["personalInfo"]["name"], "Ada")
                self.assertIn("Built reliable services", master["raw_data"])
                read = await self.client.get(self.base + "/" + master["id"])
                self.assertEqual(read.json(), master)
                self.assertEqual((await self.client.get(self.base)).json(), master)
                stats = await self.uow().get_stats(self.store.profile_id)
                self.assertTrue(stats["has_master_resume"])
                self.assertEqual(stats["total_resumes"], 1)
                self.assertEqual(self.store.profiles[self.store.profile_id].raw_data, "Original profile story")
                self.assertIsNot(self.contexts[-1], self.contexts[-2])
                retried = await self.answer(payload["thread_id"])
                self.assertEqual(retried.json(), result)
                duplicate = await self.start(method)
                self.assertEqual(duplicate.status_code, 409)
        self.assertTrue(any("FOR UPDATE" in sql for sql in self.store.statements))
        self.assertTrue(any("resumes.type" in sql for sql in self.store.statements))

    async def test_recompiled_graph_resumes_checkpointed_interview(self):
        self.llm.complete.side_effect = [turn(True), turn(False)]
        thread_id = (await self.start()).json()["thread_id"]
        self.graph = create_resume_agent(self.checkpointer)
        result = await self.answer(thread_id, "Answer after graph reconstruction")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "completed")
        self.assertIn("Answer after graph reconstruction", result.json()["master"]["raw_data"])
        self.assertEqual(len(self.store.resumes), 1)

    async def test_creation_choice_and_thread_binding(self):
        started = await self.start(None)
        payload = started.json()
        self.assertEqual(payload["interrupt"]["options"], ["manual", "profile"])
        invalid = await self.answer(payload["thread_id"], "other")
        self.assertIn("Choose", invalid.json()["interrupt"]["error"])
        foreign_id = uuid4()
        self.store.profiles[foreign_id] = Profile(id=foreign_id, raw_data="Other story")
        foreign = await self.client.post(
            f"/api/v1/resume/{foreign_id}/master/threads/{payload['thread_id']}/answers",
            json={"answer": "manual"},
        )
        self.assertEqual(foreign.status_code, 404)
        unknown = await self.answer(uuid4())
        self.assertEqual(unknown.status_code, 404)
        changed_language = await self.client.post(
            self.base + f"/threads/{payload['thread_id']}/answers",
            json={"answer": "manual", "language": "ru"},
        )
        self.assertEqual(changed_language.status_code, 422)
        self.llm.complete.return_value = turn(False)
        saved = await self.answer(payload["thread_id"], "manual")
        self.assertEqual(saved.json()["master"]["language"], "en")

    async def test_initial_analysis_failure_returns_recoverable_thread(self):
        for method in ("manual", "profile"):
            with self.subTest(method=method):
                self.store.resumes.clear()
                self.llm.complete.side_effect = [LLMError("provider_unavailable"), turn(False)]
                failed = await self.start(method)
                self.assertEqual(failed.status_code, 503)
                self.assertEqual(failed.json()["detail"], "resume_provider_failed")
                thread_id = failed.headers["X-Resume-Thread-ID"]
                self.graph = create_resume_agent(self.checkpointer)
                recovered = await self.answer(thread_id, "Retry text is not a new answer")
                self.assertEqual(recovered.status_code, 200, recovered.text)
                self.assertEqual(recovered.json()["status"], "completed")
                self.assertNotIn("Retry text", recovered.json()["master"]["raw_data"])
                self.assertEqual(len(self.store.resumes), 1)

    async def test_initial_post_commit_failure_returns_recoverable_thread(self):
        self.llm.complete.return_value = turn(False)
        original = ResumeService.save_resume

        async def crash_after_commit(service, resume):
            await original(service, resume)
            raise RuntimeError("Private post-commit failure")

        with patch.object(ResumeService, "save_resume", crash_after_commit):
            failed = await self.start()
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("Private", failed.text)
        self.assertEqual(len(self.store.resumes), 1)
        saved_id = next(iter(self.store.resumes))
        recovered = await self.answer(failed.headers["X-Resume-Thread-ID"])
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["master"]["id"], str(saved_id))
        self.assertEqual(len(self.store.resumes), 1)
        self.assertEqual(self.llm.complete.await_count, 1)

    async def test_provider_failure_retries_checkpointed_answer_once(self):
        self.llm.complete.side_effect = [turn(True), LLMError("provider_unavailable"), turn(False)]
        thread_id = (await self.start()).json()["thread_id"]
        failed = await self.answer(thread_id, "Actual answer")
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("Actual answer", failed.text)
        recovered = await self.answer(thread_id, "Retry text must not be recorded")
        self.assertEqual(recovered.status_code, 200, recovered.text)
        raw = recovered.json()["master"]["raw_data"]
        self.assertEqual(raw.count("Actual answer"), 1)
        self.assertNotIn("Retry text", raw)
        self.assertEqual(len(self.store.resumes), 1)

    async def test_save_replay_recovers_same_id_without_replacing_facts(self):
        self.llm.complete.side_effect = [turn(True), turn(False)]
        thread_id = (await self.start()).json()["thread_id"]
        original = ResumeService.save_resume
        saved_ids = []

        async def crash_after_commit(service, resume):
            result = await original(service, resume)
            saved_ids.append(result.id)
            raise RuntimeError("Simulated failure before checkpoint commit")

        with patch.object(ResumeService, "save_resume", crash_after_commit):
            failed = await self.answer(thread_id)
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(len(self.store.resumes), 1)
        first = self.store.resumes[saved_ids[0]]
        recovered = await self.answer(thread_id)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["master"]["id"], str(first.id))
        self.assertIs(self.store.resumes[first.id], first)
        self.assertEqual(len(self.store.resumes), 1)

    async def test_concurrent_turns_do_not_double_save(self):
        self.llm.complete.side_effect = [turn(True), turn(False)]
        thread_id = (await self.start()).json()["thread_id"]
        left, right = await asyncio.gather(self.answer(thread_id), self.answer(thread_id))
        self.assertEqual(left.status_code, 200, left.text)
        self.assertEqual(right.json(), left.json())
        self.assertEqual(self.llm.complete.await_count, 2)
        self.assertEqual(len(self.store.resumes), 1)

    async def test_different_threads_conflict_and_preserve_existing_master(self):
        self.llm.complete.side_effect = [turn(True), turn(True), turn(False)]
        one = (await self.start()).json()["thread_id"]
        two = (await self.start()).json()["thread_id"]
        saved = await self.answer(one)
        other = await self.answer(two)
        self.assertEqual(other.status_code, 409)
        self.assertEqual((await self.client.get(self.base)).json(), saved.json()["master"])

    async def test_import_filename_source_language_and_read(self):
        self.llm.complete.return_value = document()
        result = await self.upload(filename="../Original CV.DOCX")
        self.assertEqual(result.status_code, 201, result.text)
        master = result.json()
        self.assertEqual(master["filename"], "../Original CV.DOCX")
        self.assertEqual(master["raw_data"], "Original extracted resume")
        self.assertEqual(master["language"], "de")
        self.assertEqual((await self.client.get(self.base + "/" + master["id"])).json(), master)
        parsed_path = self.parser.parse.call_args.args[0]
        self.assertEqual(parsed_path.name, "resume.docx")
        self.assertEqual((await self.upload()).status_code, 409)
        self.assertEqual(self.parser.parse.call_count, 1)

    async def test_upload_rejections_bounded_read_and_language(self):
        for filename, content, language, status in [
            ("cv.txt", b"text", "en", 415),
            ("cv.pdf", b"", "en", 400),
            ("cv.pdf", b"x" * (MAX_UPLOAD_BYTES + 1), "en", 413),
            ("cv.pdf", b"text", "unknown", 422),
        ]:
            with self.subTest(status=status):
                self.assertEqual((await self.upload(filename, content, language)).status_code, status)
        with patch.object(UploadFile, "read", new_callable=AsyncMock, return_value=b"") as read:
            result = await self.upload()
        self.assertEqual(result.status_code, 400)
        read.assert_awaited_once_with(MAX_UPLOAD_BYTES + 1)
        self.parser.parse.assert_not_called()
        self.assertEqual(self.store.resumes, {})

    async def test_extraction_and_provider_errors_are_distinct_and_private(self):
        for error, status in [(ConversionError("private path"), 422), (RuntimeError("secret"), 503)]:
            self.parser.parse.side_effect = error
            result = await self.upload()
            self.assertEqual(result.status_code, status)
            self.assertNotIn("private", result.text)
            self.assertNotIn("secret", result.text)
        self.parser.parse.side_effect = None
        self.parser.parse.return_value = " \n"
        self.assertEqual((await self.upload()).status_code, 422)
        self.parser.parse.return_value = "Text"
        for code, status in [("invalid_structured_output", 502), ("provider_unavailable", 503)]:
            self.llm.complete.side_effect = LLMError(code)
            self.assertEqual((await self.upload()).status_code, status)
        self.assertEqual(self.store.resumes, {})

    async def test_missing_profiles_masters_and_associations(self):
        self.assertEqual((await self.client.get(self.base)).status_code, 404)
        missing = self.base.replace(str(self.store.profile_id), str(uuid4()))
        self.assertEqual((await self.client.get(missing)).status_code, 404)
        self.assertEqual((await self.client.post(missing + "/threads", json={"language": "en"})).status_code, 404)
        self.llm.complete.return_value = document()
        master = (await self.upload()).json()
        self.assertEqual((await self.client.get(self.base + "/" + str(uuid4()))).status_code, 404)
        other = uuid4()
        self.store.profiles[other] = Profile(id=other, raw_data="Other")
        foreign = f"/api/v1/resume/{other}/master/{master['id']}"
        self.assertEqual((await self.client.get(foreign)).status_code, 404)

    async def test_unique_race_maps_to_conflict(self):
        service = self.resume_service()
        class UniqueMasterViolation(Exception):
            constraint_name = "uq_resumes_profile_master_resume"

        orig = RuntimeError("private duplicate details")
        orig.__cause__ = UniqueMasterViolation()
        service._uow.resumes.create = AsyncMock(side_effect=IntegrityError("sql", {}, orig))
        with self.assertRaises(ResumeError) as caught:
            await service.save_resume(service.master_resume_from_data(
                self.store.profile_id, "Engineer", "en", document(), raw_data="Source",
            ))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertFalse(self.store.lock.locked())


class CompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_container_mappers_mounted_routes_and_password_safe_conninfo(self):
        from pydantic import SecretStr

        from src.config.settings import DatabaseSettings, Settings
        from src.main import app

        configure_mappers()
        container = create_container()
        await container.close()
        paths = app.openapi()["paths"]
        self.assertIn("/api/v1/resume/{profile_id}/master/import", paths)
        settings = Settings(database=DatabaseSettings(password=SecretStr("secret p@ss")))
        conninfo = checkpoint_conninfo(settings)
        self.assertTrue(conninfo.startswith("postgresql://"))
        self.assertNotIn("asyncpg", conninfo)
        self.assertNotIn("***", conninfo)
        self.assertIn("secret", conninfo)

    async def test_missing_libpq_is_bounded_without_schema_setup(self):
        from src.config.settings import Settings
        with patch.dict("sys.modules", {"langgraph.checkpoint.postgres.aio": None}):
            generator = AgentProvider().checkpoint_saver(Settings())
            with self.assertRaises(ResumeError) as caught:
                await anext(generator)
            self.assertEqual(caught.exception.status_code, 503)

    def test_master_index_and_raw_source_contract(self):
        table = cast(Table, Resume.__table__)
        index = next(index for index in table.indexes if index.name == "uq_resumes_profile_master_resume")
        self.assertTrue(index.unique)
        self.assertEqual(str(index.dialect_options["postgresql"]["where"]), "type = 'master'")
        resume = ResumeService.master_resume_from_data(uuid4(), "Title", "fr", document(), raw_data="Original")
        self.assertEqual(resume.type, ResumeType.MASTER)
        self.assertEqual(resume.raw_data, "Original")
        legacy = ResumeData.model_validate({"personal_info": {"name": "Ada"}, "work_experience": []})
        self.assertIn("personalInfo", legacy.model_dump())
        self.assertNotIn("personal_info", legacy.model_dump())
