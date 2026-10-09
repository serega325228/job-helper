import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
from crawlee import Request
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext

from src.config.settings import Settings
from src.exceptions.vacancy import VacancyNormalizationError, VacancyScrapingError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
from src.ports.browser_vacancy_source import BrowserVacancySource
from src.schemas.vacancy import (
    CollectionStatus,
    PreviewStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    VacancyStatus,
)
from src.services.vacancy_preview import VacancyPreviewService

type CrawlerFactory = Callable[[], AbstractAsyncContextManager[PlaywrightCrawler]]


class VacancyScrapingService:
    def __init__(
        self,
        crawler_factory: CrawlerFactory,
        unit_of_work: SqlAlchemyUnitOfWork,
        previews: VacancyPreviewService,
        hh: HhVacancySource,
        linkedin: LinkedInVacancySource,
        *,
        max_search_pages: int,
        max_previews: int,
        batch_size: int,
        concurrency: int,
        settings: Settings,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._uow = unit_of_work
        self._previews = previews
        self._settings = settings
        self._sources = {"hh": hh, "linkedin": linkedin}
        self._max_search_pages = max_search_pages
        self._max_previews = max_previews
        self._batch_size = batch_size
        self._concurrency = concurrency

    async def collect_previews(self, collection_id: UUID) -> list[UUID]:
        token = uuid4()
        async with self._uow as uow:
            collection = await uow.previews.get_collection(collection_id, lock=True)
            if collection is None or collection.status == CollectionStatus.FAILED:
                return []
            if collection.status == CollectionStatus.COMPLETED:
                return [
                    row.id
                    for row, _ in await uow.previews.get_collected(
                        collection_id=collection_id, status=PreviewStatus.PENDING_FILTER
                    )
                ]
            cutoff = datetime.now(UTC) - timedelta(
                seconds=self._settings.taskiq.lease_timeout_seconds
            )
            if (
                collection.processing_token is not None
                and collection.updated_at >= cutoff
            ):
                return []
            if collection.attempts >= self._settings.taskiq.stage_attempts:
                collection.status = CollectionStatus.FAILED
                collection.error = "Stage attempts exhausted"
                collection.processing_token = None
                return []
            collection.processing_token = token
            collection.attempts += 1
            collection.updated_at = datetime.now(UTC)
        pending_ids = set()

        async def save_batch(previews: list[VacancyPreview]) -> None:
            pending_ids.update(
                await self._previews.save_scraped(collection_id, previews, token)
            )

        try:
            await self.scrape_previews(
                self._sources[collection.source],
                collection.vacancy_page_url,
                VacancyScrapingQuery.model_validate(collection.query),
                VacancyHardFilters.model_validate(collection.hard_filters),
                save_batch,
            )
            async with self._uow as uow:
                row = await uow.previews.get_collection(collection_id, lock=True)
                if row.processing_token != token:
                    return []
                row.status = CollectionStatus.COMPLETED
                row.processing_token = None
                row.attempts = 0
                row.error = None
        except Exception as error:
            async with self._uow as uow:
                row = await uow.previews.get_collection(collection_id, lock=True)
                if row.processing_token == token:
                    row.processing_token = None
                    row.error = str(error)
                    if (
                        isinstance(error, ValueError)
                        or row.attempts >= self._settings.taskiq.stage_attempts
                    ):
                        row.status = CollectionStatus.FAILED
            raise
        return sorted(pending_ids)

    async def process_details(
        self, vacancy_ids: list[UUID]
    ) -> tuple[list[UUID], set[UUID]]:
        token = uuid4()
        async with self._uow as uow:
            vacancies = await uow.vacancies.claim_details(
                vacancy_ids, ProcessingStatus.PENDING_SCRAPE, token, self._settings
            )
            rows = await uow.previews.get_by_ids(
                [vacancy.preview_id for vacancy in vacancies]
            )
        claimed_ids = [vacancy.id for vacancy in vacancies]
        if vacancies:
            try:
                previews = {
                    preview.id: VacancyPreview.model_validate(preview)
                    for preview in rows
                }
                values = {}
                for source_name in sorted({vacancy.source for vacancy in vacancies}):
                    group = [
                        vacancy
                        for vacancy in vacancies
                        if vacancy.source == source_name
                    ]
                    raw_by_key = await self.scrape_details(
                        self._sources[source_name],
                        [previews[vacancy.preview_id] for vacancy in group],
                    )
                    for vacancy in group:
                        raw = raw_by_key[(vacancy.source, vacancy.external_id)]
                        values[vacancy.id] = (
                            {
                                "status": VacancyStatus.REMOVED,
                                "processing_status": ProcessingStatus.REMOVED,
                            }
                            if raw is None
                            else {
                                "raw_document": raw.model_dump(mode="json"),
                                "raw_payload": raw.raw_payload,
                                "description": raw.raw_text or "",
                                "processing_status": ProcessingStatus.PENDING_PARSE,
                            }
                        )
                async with self._uow as uow:
                    await uow.vacancies.finish_details(values, token)
            except Exception as error:
                async with self._uow as uow:
                    await uow.vacancies.fail_details(
                        claimed_ids,
                        token,
                        error,
                        self._settings,
                    )
                raise
        async with self._uow as uow:
            rows = await uow.vacancies.get_vacancies_by_ids(vacancy_ids)
            parsed_ids = [
                row.id
                for row in rows
                if row.processing_status == ProcessingStatus.PENDING_PARSE
            ]
            ready = await uow.vacancies.ready_batch_ids(vacancy_ids)
        return parsed_ids, ready

    async def scrape_previews(
        self,
        source: BrowserVacancySource | HhVacancySource,
        vacancy_page_url: str,
        query: VacancyScrapingQuery,
        filters: VacancyHardFilters,
        save_batch: Callable[[list[VacancyPreview]], Awaitable[None]],
    ) -> None:
        if isinstance(source, HhVacancySource):
            batch = []
            count = 0
            async for preview in source.search_previews(
                query, vacancy_page_url, max_pages=self._max_search_pages
            ):
                batch.append(preview)
                count += 1
                if len(batch) >= self._batch_size:
                    await save_batch(batch)
                    batch = []
                if count >= self._max_previews:
                    break
            if batch:
                await save_batch(batch)
            return
        seen: set[tuple[str, str]] = set()
        errors = []
        async with self._crawler_factory() as crawler:

            @crawler.router.handler("SEARCH")
            async def search_handler(context: PlaywrightCrawlingContext) -> None:
                if context.response.status in {404, 410}:
                    raise VacancyScrapingError("Search page is unavailable")
                await source.apply_filters(context.page, query, filters)
                for page_number in range(self._max_search_pages):
                    previews = await source.parse_previews(context.page)
                    if any(
                        preview.source != source.source_name for preview in previews
                    ):
                        raise VacancyScrapingError(
                            "Preview source does not match the adapter"
                        )
                    unseen = []
                    for preview in previews:
                        key = (preview.source, preview.external_id)
                        if key not in seen and len(seen) < self._max_previews:
                            seen.add(key)
                            unseen.append(preview)
                    for offset in range(0, len(unseen), self._batch_size):
                        await save_batch(unseen[offset : offset + self._batch_size])
                    if (
                        not previews
                        or len(seen) >= self._max_previews
                        or page_number + 1 == self._max_search_pages
                    ):
                        break
                    if not await source.advance_search(context.page):
                        break

            @crawler.failed_request_handler
            async def failed_handler(context, error: Exception) -> None:
                errors.append(error)

            await crawler.run([Request.from_url(vacancy_page_url, label="SEARCH")])
        if errors:
            raise VacancyScrapingError("Preview scraping failed") from errors[0]

    async def scrape_details(
        self,
        source: BrowserVacancySource | HhVacancySource,
        previews: list[VacancyPreview],
    ) -> dict[tuple[str, str], RawVacancy | None]:
        results = {}
        if isinstance(source, HhVacancySource):
            semaphore = asyncio.Semaphore(self._concurrency)

            async def fetch(preview: VacancyPreview) -> None:
                async with semaphore:
                    try:
                        results[
                            (preview.source, preview.external_id)
                        ] = await source.fetch_details(preview)
                    except httpx.HTTPStatusError as error:
                        if error.response.status_code not in {404, 410}:
                            raise
                        results[(preview.source, preview.external_id)] = None

            await asyncio.gather(*(fetch(preview) for preview in previews))
        else:
            errors = []
            async with self._crawler_factory() as crawler:

                @crawler.router.handler("DETAIL")
                async def detail_handler(context: PlaywrightCrawlingContext) -> None:
                    preview = VacancyPreview.model_validate(
                        context.request.user_data["preview"]
                    )
                    key = (preview.source, preview.external_id)
                    results[key] = (
                        None
                        if context.response.status in {404, 410}
                        else await source.parse_vacancy(context.page, preview)
                    )

                @crawler.failed_request_handler
                async def failed_handler(context, error: Exception) -> None:
                    errors.append(error)

                await crawler.run(
                    [
                        Request.from_url(
                            str(preview.url),
                            label="DETAIL",
                            user_data={"preview": preview.model_dump(mode="json")},
                        )
                        for preview in previews
                    ]
                )
            if errors:
                raise VacancyScrapingError("Detail scraping failed") from errors[0]
        for preview in previews:
            key = (preview.source, preview.external_id)
            if key not in results:
                raise VacancyScrapingError(
                    "Crawler did not fetch every requested detail"
                )
            raw = results[key]
            if raw is not None and (raw.source, raw.external_id) != key:
                raise VacancyNormalizationError(
                    "Detail identity does not match its preview"
                )
        return results
