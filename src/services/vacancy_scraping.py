import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager

import httpx
from crawlee import Request
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext

from src.exceptions.vacancy import VacancyNormalizationError, VacancyScrapingError
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.ports.browser_vacancy_source import BrowserVacancySource
from src.schemas.vacancy import (
    RawVacancy,
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
)
from src.services.vacancy_preview import VacancyPreviewService

type CrawlerFactory = Callable[[], AbstractAsyncContextManager[PlaywrightCrawler]]
type PreviewServiceFactory = Callable[
    [], AbstractAsyncContextManager[VacancyPreviewService]
]


class VacancyScrapingService:
    def __init__(
        self,
        crawler_factory: CrawlerFactory,
        *,
        max_search_pages: int,
        max_previews: int,
        batch_size: int,
        concurrency: int,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._max_search_pages = max_search_pages
        self._max_previews = max_previews
        self._batch_size = batch_size
        self._concurrency = concurrency

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
