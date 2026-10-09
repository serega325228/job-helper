import unittest
from datetime import UTC, datetime, timedelta
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from sqlalchemy.dialects import postgresql
from taskiq import TaskiqMessage, TaskiqResult
from taskiq_aio_pika.broker import AioPikaBroker

from src.config.settings import Settings
from src.di.container import create_container
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.vacancy import Vacancy, VacancyBatch
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.infrastructure.models.vacancy_preview import (
    CollectedPreview,
    PreviewCollection,
)
from src.infrastructure.models.vacancy_preview import VacancyPreview as PreviewModel
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_preview import VacancyPreviewRepository
from src.schemas.scoring import (
    LayaComparison,
    PreferenceComparison,
    ProfileComparison,
    VacancyRerankScores,
)
from src.schemas.vacancy import (
    BatchStatus,
    CollectionStatus,
    NormalizedVacancy,
    PreviewSelectionRequest,
    PreviewStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyPreview,
)
from src.schemas.vacancy_match import VacancyEmbeddingSearchResult
from src.services.scoring import ScoringService
from src.services.vacancy import VacancyRecoveryService, VacancyService
from src.services.vacancy_match import VacancyMatchService
from src.services.vacancy_preview import VacancyPreviewService
from src.services.vacancy_scraping import VacancyScrapingService
from src.tasks import scraping as tasks


async def run_task(task, *args, dependencies):
    container = SimpleNamespace(
        get=AsyncMock(
            side_effect=lambda dependency, component="": dependencies[dependency]
        )
    )
    await task.original_func(*args, dishka_container=container)


class VacancyPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.uow = AsyncMock()
        self.active = False

        async def enter():
            self.assertFalse(self.active)
            self.active = True
            return self.uow

        async def leave(*args):
            self.active = False

        self.uow.__aenter__.side_effect = enter
        self.uow.__aexit__.side_effect = leave
        self.settings = Settings(_env_file=None)
        self.dependencies = {Settings: self.settings}
        self.queues = {}
        for task in tasks.broker.get_all_tasks().values():
            mock = AsyncMock()
            self.queues[task.task_name] = mock
            patcher = patch.object(task, "kiq", mock)
            patcher.start()
            self.addCleanup(patcher.stop)

    def scraping(self, previews=None):
        return VacancyScrapingService(
            Mock(),
            self.uow,
            previews or Mock(),
            Mock(),
            Mock(),
            max_search_pages=10,
            max_previews=250,
            batch_size=32,
            concurrency=3,
            settings=self.settings,
        )

    def cohort(self, rerank_limit=1):
        now = datetime.now(UTC)
        preferences = [
            PreferenceIntent(
                id=uuid4(),
                enabled=True,
                weight=1,
                title_embedding=None,
                content_embedding=None,
            )
            for _ in range(2)
        ]
        self.batch = VacancyBatch(
            id=uuid4(),
            profile_id=uuid4(),
            preference_ids=[str(item.id) for item in preferences],
            hard_filters={},
            status=BatchStatus.DETAILS,
            search_limit=100,
            rerank_limit=rerank_limit,
            title_weight=0.4,
            attempts=0,
            updated_at=now,
        )
        self.vacancies = [
            Vacancy(
                id=UUID(int=index + 1),
                preview_id=uuid4(),
                source="hh",
                external_id=str(index),
                url=f"https://hh.ru/vacancy/{index}",
                title="AI engineer",
                company_name="Example",
                description="",
                processing_status=ProcessingStatus.PENDING_SCRAPE,
                attempts=0,
                updated_at=now,
            )
            for index in range(2)
        ]
        self.batch.vacancy_ids = [str(item.id) for item in self.vacancies]
        self.matches = []
        self.uow.vacancies.get_batch.return_value = self.batch
        self.uow.vacancies.get_stage.side_effect = lambda ids, status: [
            item
            for item in self.vacancies
            if item.id in ids and item.processing_status == status
        ]
        self.uow.vacancies.get_vacancies_by_ids.side_effect = lambda ids, **kwargs: [
            item for item in self.vacancies if item.id in ids
        ]
        for name in ("claim_details", "finish_details", "fail_details"):
            getattr(self.uow.vacancies, name).side_effect = MethodType(
                getattr(VacancyRepository, name), self.uow.vacancies
            )
        self.uow.vacancies.ready_batch_ids.side_effect = lambda ids: (
            {self.batch.id}
            if self.batch.status == BatchStatus.DETAILS
            and not any(
                item.processing_status
                in {ProcessingStatus.PENDING_SCRAPE, ProcessingStatus.PENDING_PARSE}
                for item in self.vacancies
            )
            else set()
        )
        self.uow.previews.get_by_ids.return_value = [
            PreviewModel(
                id=item.preview_id,
                source=item.source,
                external_id=item.external_id,
                title=item.title,
                url=item.url,
            )
            for item in self.vacancies
        ]
        self.uow.profiles.get_by_id.return_value = SimpleNamespace(
            id=self.batch.profile_id
        )
        self.uow.profiles.get_preferences_by_ids.return_value = preferences
        self.uow.vacancy_matches.get_batch_matches.side_effect = (
            lambda *args, **kwargs: self.matches
        )

        async def initial(values):
            self.matches.extend(VacancyMatch(id=uuid4(), **value) for value in values)

        self.uow.vacancy_matches.upsert_initial.side_effect = initial
        self.uow.vacancy_matches.embedding_scores.return_value = [
            VacancyEmbeddingSearchResult(
                vacancy_id=vacancy.id,
                preference_id=preference.id,
                title_similarity=0.9,
                content_similarity=0.8,
                combined_similarity=0.99
                if preference is preferences[0]
                else 0.86 - index * 0.1,
            )
            for index, vacancy in enumerate(self.vacancies)
            for preference in preferences
        ]
        self.scoring = Mock(spec=ScoringService)
        self.scoring.embedding_fingerprint.return_value = "fingerprint"

        async def embed(items):
            self.assertFalse(self.active)
            return {item.id: ([1.0, 0.0], [1.0, 0.0], "fingerprint") for item in items}

        async def rerank(profile, items, prefs):
            self.assertFalse(self.active)
            return {
                item.id: VacancyRerankScores(profile_score=0.9, preference_score=0.8)
                for item in items
            }

        async def evaluate(profile, prefs, items, **kwargs):
            self.assertFalse(self.active)
            self.assertTrue(
                all(
                    item.description == "Full AI requirements"
                    for item in items.values()
                )
            )
            return {
                item.id: LayaComparison(role_fit="good", skill_fit="strong")
                for item in items.values()
            }

        self.scoring.embed_entities.side_effect = embed
        self.scoring.compare_profile.return_value = ProfileComparison(score=0.8)
        self.scoring.compare_preference.side_effect = lambda pref, vac: (
            PreferenceComparison(
                score=0.8, hard_constraints_passed=pref is preferences[1]
            )
        )
        self.scoring.rerank_vacancies.side_effect = rerank
        self.scoring.laya_match_vacancies.side_effect = evaluate
        self.matching = VacancyMatchService(self.uow, self.scoring, self.settings)
        normalizer = SimpleNamespace(
            normalize=AsyncMock(
                side_effect=lambda items: [
                    NormalizedVacancy(
                        source=item.source,
                        external_id=item.external_id,
                        title=item.title,
                        description=item.raw_text,
                    )
                    for item in items
                ]
            )
        )
        parsing = VacancyService(self.uow, normalizer, self.settings)
        scraping = self.scraping()

        async def details(source, previews):
            self.assertFalse(self.active)
            return {
                (item.source, item.external_id): RawVacancy(
                    source=item.source,
                    external_id=item.external_id,
                    title=item.title,
                    url=item.url,
                    raw_text="Full AI requirements",
                    fetched_at=now,
                )
                for item in previews
            }

        scraping.scrape_details = AsyncMock(side_effect=details)
        self.dependencies.update(
            {
                VacancyMatchService: self.matching,
                VacancyService: parsing,
                VacancyScrapingService: scraping,
            }
        )
        return preferences, normalizer, scraping

    async def test_collection_persists_before_publish_and_replays_only_pending_handoff(
        self,
    ):
        collection = PreviewCollection(
            id=uuid4(),
            source="linkedin",
            status=CollectionStatus.PENDING,
            attempts=0,
            updated_at=datetime.now(UTC),
            vacancy_page_url="https://www.linkedin.com/jobs/search/",
            query={"text": "AI"},
            hard_filters={},
        )
        self.uow.previews.get_collection.return_value = collection
        identifier, events = uuid4(), []
        self.uow.previews.get_collected.return_value = [
            (SimpleNamespace(id=identifier), Mock())
        ]

        async def persist(*args):
            self.assertFalse(self.active)
            events.append("persist")
            return [identifier]

        async def scrape(source, url, query, filters, save_batch):
            self.assertFalse(self.active)
            await save_batch(
                [
                    VacancyPreview(
                        source="linkedin",
                        external_id="1",
                        title="AI",
                        url="https://www.linkedin.com/jobs/view/1/",
                    )
                ]
            )

        service = self.scraping(SimpleNamespace(save_scraped=persist))
        service.scrape_previews = AsyncMock(side_effect=scrape)
        self.queues[tasks.filter_previews.task_name].side_effect = lambda *args: (
            events.append("enqueue")
        )
        self.dependencies[VacancyScrapingService] = service
        await run_task(
            tasks.collect_previews, collection.id, dependencies=self.dependencies
        )
        await run_task(
            tasks.collect_previews, collection.id, dependencies=self.dependencies
        )
        self.assertEqual(events, ["persist", "enqueue", "enqueue"])
        self.assertEqual(collection.status, CollectionStatus.COMPLETED)
        service.scrape_previews.assert_awaited_once()

    async def test_detailed_flow_chooses_eligible_preference_and_replays_failed_publish(
        self,
    ):
        preferences, normalizer, scraping = self.cohort()
        ids = [item.id for item in self.vacancies]
        await run_task(tasks.scrape_details, ids, dependencies=self.dependencies)
        self.queues[tasks.score_embeddings.task_name].assert_not_awaited()
        await run_task(tasks.parse_vacancies, ids, dependencies=self.dependencies)
        self.queues[tasks.score_embeddings.task_name].assert_awaited_once_with(
            self.batch.id
        )
        publish = self.queues[tasks.rerank_vacancies.task_name]
        publish.side_effect = [OSError("MQ unavailable"), None]
        with self.assertRaises(OSError):
            await run_task(
                tasks.score_embeddings, self.batch.id, dependencies=self.dependencies
            )
        await run_task(
            tasks.score_embeddings, self.batch.id, dependencies=self.dependencies
        )
        self.scoring.embed_entities.assert_awaited_once()
        self.uow.vacancy_matches.upsert_initial.assert_awaited_once()
        for task in (
            tasks.rerank_vacancies,
            tasks.evaluate_vacancies,
            tasks.save_vacancy_scores,
        ):
            await run_task(task, self.batch.id, dependencies=self.dependencies)
            await run_task(task, self.batch.id, dependencies=self.dependencies)
        await run_task(tasks.scrape_details, ids, dependencies=self.dependencies)
        await run_task(tasks.parse_vacancies, ids, dependencies=self.dependencies)
        scraping.scrape_details.assert_awaited_once()
        normalizer.normalize.assert_awaited_once()
        self.scoring.rerank_vacancies.assert_awaited_once()
        self.scoring.laya_match_vacancies.assert_awaited_once()
        self.assertEqual(self.batch.status, BatchStatus.COMPLETED)
        self.assertEqual(self.matches[0].preference_intent_id, preferences[1].id)
        self.assertIsNotNone(self.matches[0].total_score)
        self.assertEqual(self.matches[1].processing_status, ProcessingStatus.FILTERED)
        self.assertNotIn("laya", self.matches[1].component_scores)
        self.assertTrue(
            all(
                item.processing_status == ProcessingStatus.COMPLETED
                for item in self.vacancies
            )
        )
        self.assertNotIn("total_score", Vacancy.__table__.columns)
        self.assertNotIn("batch_id", Vacancy.__table__.columns)

    async def test_laya_checkpoints_chunks_and_does_not_repeat_committed_evaluations(
        self,
    ):
        self.cohort(rerank_limit=2)
        ids = [item.id for item in self.vacancies]
        await self.dependencies[VacancyScrapingService].process_details(ids)
        await self.dependencies[VacancyService].parse_vacancies(ids)
        await self.matching.score_embeddings(self.batch.id)
        await self.matching.rerank_batch(self.batch.id)
        compute = self.scoring.laya_match_vacancies.side_effect
        count = 0

        async def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("model unavailable")
            return await compute(*args, **kwargs)

        self.scoring.laya_match_vacancies.side_effect = fail_second
        with self.assertRaises(OSError):
            await self.matching.evaluate_batch(self.batch.id, batch_size=1)
        self.assertEqual(
            self.matches[0].processing_status, ProcessingStatus.PENDING_SAVE
        )
        self.assertEqual(
            self.matches[1].processing_status, ProcessingStatus.PENDING_LAYA
        )
        self.assertTrue(await self.matching.evaluate_batch(self.batch.id, batch_size=1))
        calls = self.scoring.laya_match_vacancies.await_args_list
        self.assertEqual(list(calls[0].args[2]), [self.vacancies[0].id])
        self.assertEqual(list(calls[1].args[2]), [self.vacancies[1].id])
        self.assertEqual(list(calls[2].args[2]), [self.vacancies[1].id])
        await self.matching.save_batch_scores(self.batch.id)
        self.assertEqual(self.batch.status, BatchStatus.COMPLETED)

    async def test_barrier_and_recovery_are_bounded_and_do_not_duplicate_live_work(
        self,
    ):
        self.cohort()
        self.assertFalse(await self.matching.score_embeddings(self.batch.id))
        self.assertEqual(self.batch.attempts, 0)
        old = datetime.now(UTC) - timedelta(hours=2)
        preview = SimpleNamespace(
            id=uuid4(), status=PreviewStatus.PENDING_FILTER, updated_at=old
        )
        exhausted = SimpleNamespace(
            id=uuid4(),
            status=CollectionStatus.PENDING,
            attempts=3,
            processing_token=uuid4(),
            updated_at=old,
        )
        self.batch.status = BatchStatus.PENDING_SAVE
        self.batch.updated_at = old
        self.uow.previews.stale_collections.side_effect = lambda *args: (
            [exhausted] if exhausted.status == CollectionStatus.PENDING else []
        )
        self.uow.previews.stale_previews.side_effect = lambda cutoff, limit: (
            [preview] if preview.updated_at < cutoff else []
        )
        self.uow.vacancies.stale_vacancies.return_value = []
        self.uow.vacancies.stale_batches.side_effect = (
            lambda cutoff, lease_cutoff, limit: (
                [self.batch] if self.batch.updated_at < cutoff else []
            )
        )
        service = VacancyRecoveryService(self.uow)
        first = await service.claim_stale_processing(self.settings)
        self.assertEqual(first[PreviewStatus.PENDING_FILTER], [preview.id])
        self.assertEqual(exhausted.status, CollectionStatus.FAILED)
        self.assertEqual(await service.claim_stale_processing(self.settings), {})

    async def test_recent_duplicates_share_data_but_have_separate_collection_memberships(
        self,
    ):
        session = AsyncMock()
        now = datetime.now(UTC)
        canonical = PreviewModel(
            id=uuid4(),
            source="hh",
            external_id="exact",
            title="AI engineer",
            url="https://hh.ru/vacancy/exact",
            normalized_company="example",
            normalized_title="ai engineer",
            discovered_at=now - timedelta(days=1),
        )
        session.scalars.side_effect = lambda statement: (
            [canonical]
            if statement.column_descriptions[0]["entity"] is PreviewModel
            else []
        )
        session.execute.return_value = []
        repository = VacancyPreviewRepository(session)
        data = VacancyPreview(
            source="hh",
            external_id="repost",
            title=" AI   Engineer ",
            company_name=" Example ",
            url="https://hh.ru/vacancy/repost",
        )
        collections = [uuid4(), uuid4()]
        for identifier in collections:
            await repository.save_new(identifier, [data])
        inserts = [
            call.args[0].compile(dialect=postgresql.dialect())
            for call in session.execute.await_args_list
            if str(call.args[0]).startswith("INSERT INTO")
        ]
        self.assertEqual(len(inserts), 2)
        self.assertTrue(
            all(
                "INSERT INTO collected_previews" in str(statement)
                for statement in inserts
            )
        )
        self.assertEqual(
            {statement.params["collection_id_m0"] for statement in inserts},
            set(collections),
        )
        self.assertTrue(
            all(
                statement.params["preview_id_m0"] == canonical.id
                for statement in inserts
            )
        )
        selected = CollectedPreview(
            id=uuid4(),
            collection_id=collections[1],
            preview_id=canonical.id,
            status=PreviewStatus.READY,
        )
        intent = SimpleNamespace(id=uuid4(), enabled=True)
        self.uow.profiles.get_preferences_by_ids.return_value = [intent]
        collection = PreviewCollection(
            id=collections[1],
            preference_ids=[str(intent.id)],
            query={"text": "AI"},
            hard_filters={"salary_min": 300000},
        )
        self.uow.previews.get_collection.return_value = collection
        self.uow.previews.get_collected.return_value = [(selected, canonical)]
        existing = Vacancy(
            id=uuid4(),
            source=canonical.source,
            external_id=canonical.external_id,
            processing_status=ProcessingStatus.COMPLETED,
        )
        self.uow.vacancies.create_from_previews.return_value = [existing]
        self.uow.vacancies.busy_vacancy_ids.return_value = set()
        self.uow.vacancies.add_batch = Mock()
        evaluator = Mock()
        evaluator.accept.return_value = True
        service = VacancyPreviewService(self.uow, evaluator)
        self.assertEqual(
            await service.select_for_details(
                PreviewSelectionRequest(profile_id=uuid4())
            ),
            [existing.id],
        )
        batch = self.uow.vacancies.add_batch.call_args.args[0]
        self.assertEqual(batch.vacancy_ids, [str(existing.id)])
        self.assertEqual(
            batch.selection_context[str(collection.id)]["hard_filters"]["salary_min"],
            300000,
        )
        self.assertEqual(selected.status, PreviewStatus.SELECTED)
        self.assertEqual(existing.processing_status, ProcessingStatus.COMPLETED)

    async def test_single_broker_routes_consumers_and_delays_without_sleeping(self):
        broker = tasks.broker
        for task in broker.get_all_tasks().values():
            self.assertEqual(task.labels["ack_type"], "when_executed")
            self.assertEqual(
                task.labels["queue_name"],
                "laya" if task is tasks.evaluate_vacancies else "io",
            )
        message = TaskiqMessage(
            task_id=str(uuid4()),
            task_name=tasks.evaluate_vacancies.task_name,
            labels={"queue_name": "laya"},
            args=[str(uuid4())],
            kwargs={},
        )
        with patch.object(broker, "kick", new_callable=AsyncMock) as publish:
            for attempt in range(3):
                message.labels["_retries"] = attempt
                error = RuntimeError("transient")
                await broker.middlewares[0].on_error(
                    message,
                    TaskiqResult(
                        is_err=True, return_value=None, error=error, execution_time=0
                    ),
                    error,
                )
            self.assertEqual(publish.await_count, 2)
            sent = publish.await_args_list[0].args[0]
            self.assertEqual(sent.labels["queue_name"], "laya")
            self.assertGreaterEqual(float(sent.labels["_stage_delay"]), 5)
        channel = Mock()
        queues = [
            (SimpleNamespace(name=f"job-helper.{name}"), {}) for name in ("io", "laya")
        ]
        with (
            patch.object(broker, "read_channel", channel),
            patch.object(
                AioPikaBroker,
                "_declare_queues",
                new_callable=AsyncMock,
                return_value=queues,
            ),
            patch.object(self.settings.taskiq, "worker_queue", "laya"),
            patch("src.infrastructure.taskiq.broker.settings", self.settings),
        ):
            selected = await broker._declare_queues(channel)
            self.assertEqual([queue.name for queue, _ in selected], ["job-helper.laya"])
        channel = AsyncMock()
        channel.declare_queue.return_value = SimpleNamespace(name="delayed")
        with patch.object(broker, "write_channel", channel):
            await broker.kick(sent)
        arguments = channel.declare_queue.await_args.kwargs["arguments"]
        self.assertEqual(arguments["x-dead-letter-routing-key"], "laya")
        channel.default_exchange.publish.assert_awaited_once()
        container = create_container()
        await container.close()


if __name__ == "__main__":
    unittest.main()
