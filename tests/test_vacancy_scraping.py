import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from src.config.settings import Settings
from src.di.container import create_container
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy import Vacancy, VacancyBatch
from src.infrastructure.models.vacancy_preview import PreviewCollection
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
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
    PreviewStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyPreview,
)
from src.schemas.vacancy_match import VacancyEmbeddingSearchResult
from src.services.scoring import ScoringService
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService
from src.services.vacancy_scraping import PreviewServiceFactory, VacancyScrapingService
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
        self.uow.__aenter__.return_value = self.uow
        self.settings = Settings(_env_file=None)
        self.dependencies = {SqlAlchemyUnitOfWork: self.uow, Settings: self.settings}
        self.queues = {}
        for task in (
            tasks.collect_previews,
            tasks.filter_previews,
            tasks.evaluate_previews,
            tasks.scrape_details,
            tasks.parse_vacancies,
            tasks.score_embeddings,
            tasks.rerank_vacancies,
            tasks.evaluate_vacancies,
            tasks.save_vacancy_scores,
        ):
            mock = AsyncMock()
            self.queues[task.task_name] = mock
            patcher = patch.object(task, "kiq", mock)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_collection_commits_previews_before_scheduling_and_skips_replay(self):
        collection = PreviewCollection(
            id=uuid4(),
            source="linkedin",
            status=CollectionStatus.PENDING,
            vacancy_page_url="https://www.linkedin.com/jobs/search/",
            query={"text": "Python"},
            hard_filters={},
        )
        self.uow.previews.get_collection.return_value = collection
        events = []
        preview_id = uuid4()

        async def persist(collection_id, previews):
            events.append("persist")
            return {
                PreviewStatus.PENDING_FILTER: [preview_id],
                PreviewStatus.PENDING_LAYA: [],
            }

        @asynccontextmanager
        async def factory():
            yield SimpleNamespace(save_scraped=persist)

        async def scrape(source, url, query, filters, save_batch):
            await save_batch(
                [
                    VacancyPreview(
                        source="linkedin",
                        external_id="42",
                        title="Python",
                        url="https://www.linkedin.com/jobs/view/42/",
                    )
                ]
            )

        self.dependencies.update(
            {
                PreviewServiceFactory: factory,
                VacancyScrapingService: SimpleNamespace(
                    scrape_previews=AsyncMock(side_effect=scrape)
                ),
                HhVacancySource: object(),
                LinkedInVacancySource: object(),
            }
        )
        self.queues[tasks.filter_previews.task_name].side_effect = lambda *args: (
            events.append("enqueue")
        )
        await run_task(
            tasks.collect_previews, collection.id, dependencies=self.dependencies
        )
        await run_task(
            tasks.collect_previews, collection.id, dependencies=self.dependencies
        )
        self.assertEqual(events, ["persist", "enqueue"])
        self.assertEqual(collection.status, CollectionStatus.COMPLETED)
        self.dependencies[VacancyScrapingService].scrape_previews.assert_awaited_once()

    async def test_detailed_stages_reload_shortlist_rerank_evaluate_and_skip_replays(
        self,
    ):
        preference_id = uuid4()
        batch = VacancyBatch(
            id=uuid4(),
            profile_id=uuid4(),
            preference_ids=[str(preference_id)],
            hard_filters={},
            status=BatchStatus.DETAILS,
            search_limit=100,
            rerank_limit=1,
            title_weight=0.4,
        )
        vacancies = [
            Vacancy(
                id=uuid4(),
                batch_id=batch.id,
                preview_id=uuid4(),
                source="hh",
                external_id=str(index),
                url=f"https://hh.ru/vacancy/{index}",
                title="Python Developer",
                company_name="Example",
                description="",
                processing_status=ProcessingStatus.PENDING_SCRAPE,
            )
            for index in range(2)
        ]
        raw = {
            (vacancy.source, vacancy.external_id): RawVacancy(
                source=vacancy.source,
                external_id=vacancy.external_id,
                url=vacancy.url,
                title=vacancy.title,
                raw_text="Full description with Python requirements",
                fetched_at=datetime.now(UTC),
            )
            for vacancy in vacancies
        }
        previews = [
            SimpleNamespace(
                id=vacancy.preview_id,
                **VacancyPreview(
                    source=vacancy.source,
                    external_id=vacancy.external_id,
                    title=vacancy.title,
                    url=vacancy.url,
                ).model_dump(),
                evaluation={"role_fit": "strong", "skill_fit": "strong"},
            )
            for vacancy in vacancies
        ]
        self.uow.previews.get_by_ids.return_value = previews
        self.uow.vacancies.get_stage.side_effect = lambda ids, status: [
            vacancy
            for vacancy in vacancies
            if vacancy.id in ids and vacancy.processing_status == status
        ]
        self.uow.vacancies.get_batch.return_value = batch
        self.uow.vacancies.get_batch_vacancies.return_value = vacancies
        self.uow.profiles.get_by_id.return_value = SimpleNamespace(id=batch.profile_id)
        self.uow.profiles.get_preferences_by_ids.return_value = [
            SimpleNamespace(
                id=preference_id,
                enabled=True,
                title_embedding=None,
                content_embedding=None,
            )
        ]
        self.uow.vacancy_matches.get_by_profile_and_vacancy_ids.return_value = []
        self.uow.vacancy_matches.add = Mock()
        self.uow.vacancy_matches.search_by_preferences.return_value = [
            VacancyEmbeddingSearchResult(
                vacancy_id=vacancy.id,
                preference_id=preference_id,
                title_similarity=0.9,
                content_similarity=0.8,
                combined_similarity=0.86 - index * 0.1,
            )
            for index, vacancy in enumerate(vacancies)
        ]
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
        service = VacancyService(
            self.uow,
            normalizer,
            SimpleNamespace(embed=AsyncMock(return_value=[[1.0, 0.0]] * 4)),
        )
        events = []

        async def rerank(profile, items, preferences):
            events.append("rerank")
            self.assertEqual(items, [vacancies[0]])
            return {
                vacancies[0].id: VacancyRerankScores(
                    profile_score=0.9, preference_score=0.8
                )
            }

        async def evaluate(profile, preferences, items, **kwargs):
            events.append("laya")
            self.assertEqual(
                items[vacancies[0].id].description, raw[("hh", "0")].raw_text
            )
            return {
                vacancy.id: LayaComparison(role_fit="good", skill_fit="weak")
                for vacancy in vacancies
            }

        scoring = Mock(spec=ScoringService)
        scoring.compare_profile.return_value = ProfileComparison(score=0.8)
        scoring.compare_preference.return_value = PreferenceComparison(
            score=0.8, hard_constraints_passed=True
        )
        scoring.rerank_vacancies.side_effect = rerank
        scoring.laya_match_vacancies.side_effect = evaluate
        self.dependencies.update(
            {
                VacancyService: service,
                ScoringService: scoring,
                VacancyMatchService: VacancyMatchService(self.uow, scoring),
                VacancyScrapingService: SimpleNamespace(
                    scrape_details=AsyncMock(return_value=raw)
                ),
                HhVacancySource: object(),
                LinkedInVacancySource: object(),
            }
        )
        ids = [vacancy.id for vacancy in vacancies]
        for task, arguments, expected in (
            (tasks.scrape_details, [ids], ProcessingStatus.PENDING_PARSE),
            (tasks.parse_vacancies, [ids], ProcessingStatus.PENDING_EMBEDDING),
            (tasks.score_embeddings, [batch.id], ProcessingStatus.PENDING_RERANK),
            (tasks.rerank_vacancies, [batch.id], ProcessingStatus.PENDING_LAYA),
            (tasks.evaluate_vacancies, [batch.id], ProcessingStatus.PENDING_SAVE),
            (tasks.save_vacancy_scores, [batch.id], ProcessingStatus.COMPLETED),
        ):
            await run_task(task, *arguments, dependencies=self.dependencies)
            self.assertEqual(vacancies[0].processing_status, expected)
            await run_task(task, *arguments, dependencies=self.dependencies)
        self.assertEqual(events, ["rerank", "laya"])
        self.assertEqual(vacancies[1].processing_status, ProcessingStatus.FILTERED)
        self.assertEqual(
            vacancies[1].evaluation, {"role_fit": "good", "skill_fit": "weak"}
        )
        self.assertEqual(
            vacancies[0].evaluation, {"role_fit": "good", "skill_fit": "weak"}
        )
        self.assertEqual(batch.status, BatchStatus.COMPLETED)
        self.uow.vacancy_matches.add.assert_called_once()
        self.assertIn("total_score", vacancies[0].scores)
        self.assertEqual(
            self.uow.vacancy_matches.search_by_preferences.await_args.kwargs[
                "vacancy_ids"
            ],
            ids,
        )

    async def test_embedding_waits_for_the_whole_persisted_batch(self):
        batch = SimpleNamespace(id=uuid4(), status=BatchStatus.DETAILS)
        self.uow.vacancies.get_batch.return_value = batch
        self.uow.vacancies.get_batch_vacancies.return_value = [
            SimpleNamespace(processing_status=ProcessingStatus.PENDING_PARSE)
        ]
        self.dependencies.update({VacancyService: Mock(), ScoringService: Mock()})
        await run_task(tasks.score_embeddings, batch.id, dependencies=self.dependencies)
        self.uow.profiles.get_by_id.assert_not_awaited()
        self.queues[tasks.rerank_vacancies.task_name].assert_not_awaited()

    async def test_recovery_reschedules_stale_rows_once_per_timeout(self):
        old = datetime.now(UTC) - timedelta(hours=1)
        preview = SimpleNamespace(
            id=uuid4(), status=PreviewStatus.PENDING_LAYA, updated_at=old
        )
        batch = SimpleNamespace(
            id=uuid4(), status=BatchStatus.PENDING_SAVE, updated_at=old
        )
        self.uow.previews.stale_collections.return_value = []
        self.uow.vacancies.stale_vacancies.return_value = []
        self.uow.previews.stale_previews.side_effect = lambda cutoff, limit: (
            [preview] if preview.updated_at < cutoff else []
        )
        self.uow.vacancies.stale_batches.side_effect = lambda cutoff, limit: (
            [batch] if batch.updated_at < cutoff else []
        )
        await run_task(tasks.reconcile_processing, dependencies=self.dependencies)
        await run_task(tasks.reconcile_processing, dependencies=self.dependencies)
        self.queues[tasks.evaluate_previews.task_name].assert_awaited_once_with(
            [preview.id]
        )
        self.queues[tasks.save_vacancy_scores.task_name].assert_awaited_once_with(
            batch.id
        )

    async def test_duplicate_detection_rejects_exact_and_recent_reposts_but_allows_old_reposts(
        self,
    ):
        session = AsyncMock()
        now = datetime.now(UTC)
        stored = [
            ("hh", "exact", "old company", "python", now - timedelta(days=8)),
            ("hh", "recent", "example", "python developer", now - timedelta(days=1)),
        ]

        async def execute(statement):
            sql = str(statement.compile(dialect=postgresql.dialect()))
            return stored if "FROM vacancy_previews" in sql else []

        session.execute.side_effect = execute
        session.scalars.return_value = []
        repository = VacancyPreviewRepository(session)
        previews = [
            VacancyPreview(
                source="hh",
                external_id=external_id,
                title=title,
                company_name=company,
                url=f"https://hh.ru/vacancy/{external_id}",
            )
            for external_id, title, company in (
                ("exact", "Unrelated", "Elsewhere"),
                ("repost", " PYTHON   Developer ", " Example "),
                ("old-repost", "Python", "Old Company"),
            )
        ]
        await repository.save_new(uuid4(), previews, description_min_length=80)
        params = (
            session.scalars.await_args.args[0]
            .compile(dialect=postgresql.dialect())
            .params
        )
        self.assertEqual(params["external_id_m0"], "old-repost")
        self.assertNotIn("external_id_m1", params)

    async def test_workers_use_separate_queues_acknowledgement_and_valid_dependencies(
        self,
    ):
        for task in tasks.io_broker.get_all_tasks().values():
            self.assertEqual(task.labels["ack_type"], "when_executed")
        self.assertEqual(
            set(tasks.laya_broker.get_all_tasks()),
            {tasks.evaluate_previews.task_name, tasks.evaluate_vacancies.task_name},
        )
        self.assertEqual(
            tasks.io_broker._task_queues[0].routing_key, self.settings.taskiq.io_queue
        )
        self.assertEqual(
            tasks.laya_broker._task_queues[0].routing_key,
            self.settings.taskiq.laya_queue,
        )
        container = create_container()
        await container.close()


if __name__ == "__main__":
    unittest.main()
