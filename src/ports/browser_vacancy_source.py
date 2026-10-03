from typing import Protocol

from playwright.async_api import Page

from src.schemas.vacancy import (
    RawVacancy,
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
)


class BrowserVacancySource(Protocol):
    source_name: str

    def search_url(
        self, query: VacancyScrapingQuery, filters: VacancyHardFilters
    ) -> str: ...

    async def apply_filters(
        self, page: Page, query: VacancyScrapingQuery, filters: VacancyHardFilters
    ) -> None: ...

    async def parse_previews(self, page: Page) -> list[VacancyPreview]: ...

    async def advance_search(self, page: Page) -> bool: ...

    async def parse_vacancy(
        self, page: Page, preview: VacancyPreview
    ) -> RawVacancy: ...
