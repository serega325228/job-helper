import unittest
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from crawlee.errors import UserHandlerTimeoutError
from crawlee.router import Router
from pydantic import ValidationError

from src.config.settings import ScrapingSettings
from src.di.container import create_container
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.schemas.vacancy import RawVacancy, VacancyPreview, VacancyScrapingQuery
from src.services.vacancy_scraping import VacancyScrapingService


def preview(external_id, *, title="Python Developer", url=None):
    return VacancyPreview(
        source="test",
        external_id=external_id,
        title=title,
        url=url or f"https://jobs.example/vacancies/{external_id}",
    )


class FakeCrawler:
    instances: ClassVar[list[FakeCrawler]] = []
    statuses: ClassVar[dict[str, int]] = {}

    def __init__(self, **options):
        self.options = options
        self.router = Router()
        self.requests = []
        self.attempts = Counter()
        self.known = set()
        self.pending = []
        self.failed_handler = None
        self.queue = SimpleNamespace(drop=AsyncMock())
        self.store = SimpleNamespace(drop=AsyncMock())
        self.instances.append(self)

    def failed_request_handler(self, handler):
        self.failed_handler = handler
        return handler

    async def add_requests(self, requests):
        for request in requests:
            if request.unique_key not in self.known:
                self.known.add(request.unique_key)
                self.requests.append(request)
                self.pending.append(request)

    async def run(self, requests):
        await self.add_requests(requests)
        handled = 0
        while self.pending and handled < self.options["max_requests_per_crawl"]:
            request = self.pending.pop(0)
            while True:
                deferred = []

                async def add_deferred(requests, destination=deferred):
                    destination.extend(requests)

                self.attempts[request.url] += 1
                context = SimpleNamespace(
                    request=request,
                    page=SimpleNamespace(index=0),
                    response=SimpleNamespace(
                        status=self.statuses.get(request.url, 200)
                    ),
                    add_requests=add_deferred,
                )
                try:
                    await self.router(context)
                except (RuntimeError, UserHandlerTimeoutError) as error:
                    if request.retry_count < self.options["max_request_retries"]:
                        request.retry_count += 1
                        continue
                    await self.failed_handler(context, error)
                else:
                    await self.add_requests(deferred)
                break
            handled += 1

    async def get_request_manager(self):
        return self.queue

    async def get_key_value_store(self):
        return self.store


class FakeSource:
    source_name = "test"

    def __init__(self, pages, *, advance_failures=0, wrong_identity=False):
        self.pages = pages
        self.advance_failures = advance_failures
        self.wrong_identity = wrong_identity
        self.details = []
        self.filters_applied = 0

    def search_url(self, query, filters):
        return "https://jobs.example/search"

    async def apply_filters(self, page, query, filters):
        self.filters_applied += 1

    async def parse_previews(self, page):
        return self.pages[page.index]

    async def advance_search(self, page):
        if self.advance_failures:
            self.advance_failures -= 1
            raise TimeoutError("Pagination timed out")
        page.index += 1
        return page.index < len(self.pages)

    async def parse_vacancy(self, page, item):
        self.details.append(item.external_id)
        return RawVacancy(
            source=item.source,
            external_id="unexpected" if self.wrong_identity else item.external_id,
            url=item.url,
            title=item.title,
            raw_text="Build Python services",
            fetched_at=datetime.now(UTC),
        )


class FakeEvaluator:
    def __init__(self, selected=None):
        self.selected = selected
        self.batches = []

    def accept(self, item, profile, preferences, filters, query):
        return item.title != "Designer"

    async def evaluate(self, items, profile, preferences, query, *, batch_size):
        self.batches.append([item.external_id for item in items])
        return [
            item
            for item in items
            if self.selected is None or item.external_id in self.selected
        ]


class FakeUnitOfWork:
    def __init__(self, container):
        self.profiles = SimpleNamespace(
            get_by_id=AsyncMock(return_value=container.profile),
            get_preferences_by_profile_id=AsyncMock(return_value=[]),
        )
        self.vacancies = SimpleNamespace(
            get_by_external_keys=AsyncMock(side_effect=container.existing)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeContainer:
    def __init__(self):
        self.profile = SimpleNamespace(id=uuid4())
        self.items = []
        self.scopes = []
        self.normalization_failures = 0

    def __call__(self):
        container = self

        class Scope:
            async def __aenter__(self):
                self.uow = FakeUnitOfWork(container)
                self.service = SimpleNamespace(
                    normalize_vacancies=AsyncMock(side_effect=container.normalize),
                    save_vacancies=AsyncMock(side_effect=container.save),
                )
                container.scopes.append(self)
                return self

            async def __aexit__(self, *args):
                return None

            async def get(self, dependency):
                return self.uow if dependency is SqlAlchemyUnitOfWork else self.service

        return Scope()

    async def existing(self, keys, *, urls):
        return [
            item
            for item in self.items
            if (item.source, item.external_id) in keys or item.url in urls
        ]

    async def normalize(self, raw):
        if self.normalization_failures:
            self.normalization_failures -= 1
            raise RuntimeError("Temporary normalization failure")
        return raw

    async def save(self, raw, normalized):
        saved = [
            SimpleNamespace(
                id=uuid4(),
                source=item.source,
                external_id=item.external_id,
                url=str(item.url),
            )
            for item in raw
        ]
        self.items.extend(saved)
        return saved


class VacancyScrapingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeCrawler.instances = []
        FakeCrawler.statuses = {}
        self.container = FakeContainer()
        self.evaluator = FakeEvaluator()
        self.settings = ScrapingSettings(_env_file=None)
        self.crawler_patch = patch(
            "src.services.vacancy_scraping.PlaywrightCrawler", FakeCrawler
        )
        self.crawler_patch.start()
        self.addCleanup(self.crawler_patch.stop)

    async def scrape(self, source):
        service = VacancyScrapingService(
            SimpleNamespace(scraping=self.settings), self.container, self.evaluator
        )
        return await service.scrape(
            source, VacancyScrapingQuery(text="Python"), self.container.profile.id
        )

    async def test_only_filtered_laya_matches_reach_details_and_storage(self):
        self.container.items = [
            SimpleNamespace(
                source="test", external_id="known", url="https://jobs.example/old"
            ),
            SimpleNamespace(
                source="other", external_id="stored", url="https://jobs.example/stored"
            ),
        ]
        match = preview("match")
        self.evaluator.selected = {"match"}
        source = FakeSource(
            [
                [
                    preview("known"),
                    preview("alias", url="https://jobs.example/stored"),
                    preview("designer", title="Designer"),
                    preview("weak"),
                    match,
                    match,
                    preview("same-url", url=str(match.url)),
                ]
            ]
        )

        result = await self.scrape(source)

        self.assertEqual(self.evaluator.batches, [["weak", "match"]])
        self.assertEqual(source.details, ["match"])
        self.assertEqual(result.previews_discovered, 5)
        self.assertEqual(result.duplicates_skipped, 4)
        self.assertEqual(result.rejected_by_filters, 1)
        self.assertEqual(result.sent_to_laya, 2)
        self.assertEqual(result.rejected_by_laya, 1)
        self.assertEqual(result.vacancies_saved, 1)
        detail = next(
            request
            for request in FakeCrawler.instances[0].requests
            if request.label == "DETAIL"
        )
        self.assertEqual(
            VacancyPreview.model_validate(detail.user_data["preview"]), match
        )

    async def test_search_retry_preserves_accepted_previews_without_repeating_evaluation(
        self,
    ):
        source = FakeSource(
            [[preview("first")], [preview("second")]], advance_failures=1
        )

        result = await self.scrape(source)

        self.assertEqual(source.details, ["first", "second"])
        self.assertEqual(self.evaluator.batches, [["first"], ["second"]])
        self.assertEqual(result.search_pages, 2)
        self.assertEqual(result.previews_discovered, 2)
        self.assertEqual(result.vacancies_saved, 2)
        self.assertEqual(result.failed_pages, 0)

    async def test_terminal_search_failure_salvages_accepted_batch(self):
        source = FakeSource([[preview("first")], []], advance_failures=10)

        with self.assertLogs("src.services.vacancy_scraping", level="ERROR"):
            result = await self.scrape(source)

        self.assertEqual(source.details, ["first"])
        self.assertEqual(self.evaluator.batches, [["first"]])
        self.assertEqual(result.failed_pages, 1)
        self.assertEqual(result.vacancies_saved, 1)

    async def test_detail_retry_uses_fresh_scopes_and_does_not_double_count_scraped_page(
        self,
    ):
        self.container.normalization_failures = 1
        source = FakeSource([[preview("first"), preview("second")]])

        result = await self.scrape(source)

        service_scopes = [
            scope
            for scope in self.container.scopes
            if scope.service.normalize_vacancies.await_count
        ]
        self.assertEqual(len(service_scopes), 3)
        self.assertEqual(
            len({id(scope.uow) for scope in self.container.scopes}),
            len(self.container.scopes),
        )
        self.assertEqual(source.details, ["first", "first", "second"])
        self.assertEqual(result.detail_pages_scraped, 2)
        self.assertEqual(result.vacancies_saved, 2)

    async def test_terminal_detail_failure_does_not_stop_other_details(self):
        self.settings = self.settings.model_copy(update={"max_request_retries": 0})
        self.container.normalization_failures = 1
        source = FakeSource([[preview("first"), preview("second")]])

        with self.assertLogs("src.services.vacancy_scraping", level="ERROR"):
            result = await self.scrape(source)

        self.assertEqual(result.failed_pages, 1)
        self.assertEqual(result.vacancies_saved, 1)
        self.assertEqual(self.container.items[0].external_id, "second")

    async def test_removed_details_are_not_parsed_or_retried(self):
        removed = [preview("404"), preview("410")]
        FakeCrawler.statuses = {
            str(item.url): int(item.external_id) for item in removed
        }

        result = await self.scrape(FakeSource([removed]))

        self.assertEqual(result.removed_pages, 2)
        self.assertEqual(result.detail_pages_scraped, 0)
        self.assertEqual(result.vacancies_saved, 0)
        self.assertEqual(result.failed_pages, 0)
        self.assertTrue(
            all(amount == 1 for amount in FakeCrawler.instances[0].attempts.values())
        )

    async def test_detail_with_changed_identity_is_rejected(self):
        self.settings = self.settings.model_copy(update={"max_request_retries": 0})

        with self.assertLogs("src.services.vacancy_scraping", level="ERROR"):
            result = await self.scrape(
                FakeSource([[preview("first")]], wrong_identity=True)
            )

        self.assertEqual(result.failed_pages, 1)
        self.assertEqual(result.vacancies_saved, 0)
        self.assertEqual(self.container.items, [])

    async def test_page_preview_and_detail_limits_stop_discovery(self):
        for limits, expected_details, expected_pages, expected_previews in [
            ({"max_search_pages": 1}, 2, 1, 2),
            ({"max_previews": 3}, 3, 2, 3),
            ({"max_detail_pages": 1}, 1, 1, 2),
        ]:
            with self.subTest(limits=limits):
                self.container = FakeContainer()
                self.evaluator = FakeEvaluator()
                self.settings = ScrapingSettings(_env_file=None, **limits)
                source = FakeSource(
                    [
                        [preview("1"), preview("2")],
                        [preview("3"), preview("4")],
                        [preview("5")],
                    ]
                )
                result = await self.scrape(source)
                self.assertEqual(len(source.details), expected_details)
                self.assertEqual(result.search_pages, expected_pages)
                self.assertEqual(result.previews_discovered, expected_previews)

    async def test_runs_have_isolated_storage_counters_and_crawlee_configuration(self):
        first = await self.scrape(FakeSource([[preview("first")]]))
        second = await self.scrape(FakeSource([[preview("first")]]))
        options = FakeCrawler.instances[0].options

        self.assertEqual(first.vacancies_saved, 1)
        self.assertEqual(second.vacancies_saved, 0)
        self.assertEqual(second.duplicates_skipped, 1)
        self.assertEqual(second.detail_pages_enqueued, 0)
        directories = [
            crawler.options["configuration"].storage_dir
            for crawler in FakeCrawler.instances
        ]
        self.assertEqual(len(set(directories)), 2)
        self.assertTrue(all(not Path(directory).exists() for directory in directories))
        for crawler in FakeCrawler.instances:
            crawler.queue.drop.assert_awaited_once()
            crawler.store.drop.assert_awaited_once()
        self.assertFalse(options["configure_logging"])
        self.assertFalse(options["retry_on_blocked"])
        self.assertEqual(
            options["max_request_retries"], self.settings.max_request_retries
        )
        self.assertEqual(options["max_session_rotations"], 0)
        self.assertEqual(options["ignore_http_error_status_codes"], [404, 410])
        self.assertEqual(
            options["navigation_timeout"],
            timedelta(seconds=self.settings.navigation_timeout_seconds),
        )
        self.assertEqual(
            options["concurrency_settings"].max_tasks_per_minute,
            self.settings.max_requests_per_minute,
        )
        self.assertEqual(
            options["concurrency_settings"].max_concurrency,
            self.settings.max_concurrency,
        )
        self.assertEqual(options["goto_options"], {"wait_until": "domcontentloaded"})

    async def test_dependency_graph_validates_without_initializing_models(self):
        container = create_container()
        await container.close()

    def test_invalid_scraping_limits_and_concurrency_fail_fast(self):
        for values in [
            {"min_concurrency": 4},
            {"desired_concurrency": 6},
            {"max_previews": 0},
            {"max_search_pages": 0},
            {"max_detail_pages": 0},
            {"laya_batch_size": 0},
            {"max_requests_per_minute": 0},
            {"max_request_retries": -1},
        ]:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ScrapingSettings(_env_file=None, **values)


if __name__ == "__main__":
    unittest.main()
