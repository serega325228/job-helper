from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from uuid import UUID

from crawlee import Request
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from structlog import get_logger

from src.exceptions.profile import ProfileNotFoundError
from src.exceptions.vacancy import VacancyNormalizationError, VacancyScrapingError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.ports.browser_vacancy_source import BrowserVacancySource
from src.schemas.vacancy import (
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    VacancyScrapingResult,
)
from src.services.vacancy import VacancyService
from src.services.vacancy_preview import VacancyPreviewEvaluator

type CrawlerFactory = Callable[[], AbstractAsyncContextManager[PlaywrightCrawler]]
type UnitOfWorkFactory = Callable[[], AbstractAsyncContextManager[SqlAlchemyUnitOfWork]]
type VacancyServiceFactory = Callable[[], AbstractAsyncContextManager[VacancyService]]

logger = get_logger()


class VacancyScrapingService:
    def __init__(
        self,
        crawler_factory: CrawlerFactory,
        unit_of_work_factory: UnitOfWorkFactory,
        vacancy_service_factory: VacancyServiceFactory,
        evaluator: VacancyPreviewEvaluator,
        *,
        max_search_pages: int,
        max_previews: int,
        max_detail_pages: int,
        laya_batch_size: int,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._unit_of_work_factory = unit_of_work_factory
        self._vacancy_service_factory = vacancy_service_factory
        self._evaluator = evaluator
        self._max_search_pages = max_search_pages
        self._max_previews = max_previews
        self._max_detail_pages = max_detail_pages
        self._laya_batch_size = laya_batch_size

    async def _existing(self, previews: list[VacancyPreview]) -> set[tuple[str, str]]:
        async with self._unit_of_work_factory() as unit_of_work, unit_of_work as uow:
            vacancies = await uow.vacancies.get_by_external_keys(
                {(preview.source, preview.external_id) for preview in previews},
                urls={str(preview.url) for preview in previews},
            )
        keys = {(vacancy.source, vacancy.external_id) for vacancy in vacancies}
        urls = {vacancy.url for vacancy in vacancies}
        return {
            (preview.source, preview.external_id)
            for preview in previews
            if (preview.source, preview.external_id) in keys or str(preview.url) in urls
        }

    async def scrape(
        self,
        source: BrowserVacancySource,
        query: VacancyScrapingQuery,
        profile_id: UUID,
        *,
        filters: VacancyHardFilters | None = None,
    ) -> VacancyScrapingResult:
        filters = filters or VacancyHardFilters()
        result = VacancyScrapingResult(source=source.source_name)
        async with self._unit_of_work_factory() as unit_of_work, unit_of_work as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is None:
                raise ProfileNotFoundError(profile_id)
            preferences = [
                preference
                for preference in await uow.profiles.get_preferences_by_profile_id(
                    profile_id
                )
                if preference.enabled
            ]

        processed_keys: set[tuple[str, str]] = set()
        processed_urls: set[str] = set()
        planned: dict[tuple[str, str], VacancyPreview] = {}
        page_signatures: set[tuple[tuple[str, str], ...]] = set()
        scraped_keys: set[tuple[str, str]] = set()

        def detail_requests() -> list[Request]:
            return [
                Request.from_url(
                    str(preview.url),
                    label="DETAIL",
                    user_data={"preview": preview.model_dump(mode="json")},
                )
                for preview in planned.values()
            ]

        async def process_batch(batch: list[VacancyPreview]) -> None:
            existing = await self._existing(batch)
            candidates = [
                preview
                for preview in batch
                if (preview.source, preview.external_id) not in existing
                and self._evaluator.accept(
                    preview, profile, preferences, filters, query
                )
            ]
            selected = (
                await self._evaluator.evaluate(
                    candidates,
                    profile,
                    preferences,
                    query,
                    batch_size=self._laya_batch_size,
                )
                if candidates
                else []
            )
            remaining = self._max_detail_pages - len(planned)
            for preview in selected[:remaining]:
                planned[(preview.source, preview.external_id)] = preview
            processed_keys.update(
                (preview.source, preview.external_id) for preview in batch
            )
            processed_urls.update(str(preview.url) for preview in batch)
            result.previews_discovered += len(batch)
            result.duplicates_skipped += len(existing)
            result.rejected_by_filters += len(batch) - len(existing) - len(candidates)
            result.sent_to_laya += len(candidates)
            result.rejected_by_laya += len(candidates) - len(selected)

        async with self._crawler_factory() as crawler:

            @crawler.router.handler("SEARCH")
            async def search_handler(context: PlaywrightCrawlingContext) -> None:
                if context.response.status in {404, 410}:
                    raise VacancyScrapingError("Search page is unavailable")
                if (
                    len(planned) < self._max_detail_pages
                    and len(processed_keys) < self._max_previews
                ):
                    await source.apply_filters(context.page, query, filters)
                    for _page_number in range(self._max_search_pages):
                        previews = await source.parse_previews(context.page)
                        if any(
                            preview.source != source.source_name for preview in previews
                        ):
                            raise VacancyScrapingError(
                                "Preview source does not match the adapter"
                            )
                        signature = tuple(
                            (preview.source, preview.external_id)
                            for preview in previews
                        )
                        if (
                            signature not in page_signatures
                            and result.search_pages >= self._max_search_pages
                        ):
                            break
                        batch_by_key: dict[tuple[str, str], VacancyPreview] = {}
                        batch_urls: set[str] = set()
                        for preview in previews:
                            key = (preview.source, preview.external_id)
                            url = str(preview.url)
                            if (
                                key in processed_keys
                                or url in processed_urls
                                or key in batch_by_key
                                or url in batch_urls
                            ):
                                result.duplicates_skipped += 1
                                continue
                            batch_by_key[key] = preview
                            batch_urls.add(url)
                        remaining = self._max_previews - len(processed_keys)
                        unseen = list(batch_by_key.values())[:remaining]
                        for offset in range(0, len(unseen), self._laya_batch_size):
                            await process_batch(
                                unseen[offset : offset + self._laya_batch_size]
                            )
                            if len(planned) >= self._max_detail_pages:
                                break
                        if signature not in page_signatures:
                            page_signatures.add(signature)
                            result.search_pages += 1
                        if (
                            not previews
                            or len(planned) >= self._max_detail_pages
                            or len(processed_keys) >= self._max_previews
                            or _page_number + 1 >= self._max_search_pages
                        ):
                            break
                        if not await source.advance_search(context.page):
                            break
                await context.add_requests(detail_requests())
                result.detail_pages_enqueued = len(planned)

            @crawler.router.handler("DETAIL")
            async def detail_handler(context: PlaywrightCrawlingContext) -> None:
                preview = VacancyPreview.model_validate(
                    context.request.user_data["preview"]
                )
                if context.response.status in {404, 410}:
                    result.removed_pages += 1
                    logger.info(
                        "Vacancy removed source=%s url=%s request_type=DETAIL",
                        source.source_name,
                        context.request.url,
                    )
                    return
                if await self._existing([preview]):
                    result.duplicates_skipped += 1
                    return
                raw = await source.parse_vacancy(context.page, preview)
                key = (preview.source, preview.external_id)
                if (raw.source, raw.external_id) != key:
                    raise VacancyNormalizationError(
                        "Detail identity does not match its preview"
                    )
                if key not in scraped_keys:
                    scraped_keys.add(key)
                    result.detail_pages_scraped += 1
                async with self._vacancy_service_factory() as vacancy_service:
                    normalized = await vacancy_service.normalize_vacancies([raw])
                    saved = await vacancy_service.save_vacancies([raw], normalized)
                result.vacancy_ids.extend(vacancy.id for vacancy in saved)
                result.vacancies_saved += len(saved)

            @crawler.failed_request_handler
            async def failed_handler(context, error: Exception) -> None:
                result.failed_pages += 1
                logger.error(
                    "Scraping failed source=%s url=%s request_type=%s: %r",
                    source.source_name,
                    context.request.url,
                    context.request.label,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
                if context.request.label == "SEARCH" and planned:
                    await crawler.add_requests(detail_requests())
                    result.detail_pages_enqueued = len(planned)

            await crawler.run(
                [Request.from_url(source.search_url(query, filters), label="SEARCH")]
            )

        logger.info("Vacancy crawl completed: %s", result.model_dump(mode="json"))
        return result
