from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.schemas.vacancy import (
    RawVacancy,
    VacancyPreview,
    VacancyReference,
    VacancySearchQuery,
)


class HhVacancySource:
    source_name = "hh"

    def __init__(self, client: HhApiClient) -> None:
        self._client = client

    def search_url(self, query, filters) -> str:
        return "https://api.hh.ru/vacancies"

    @staticmethod
    def validate_search_url(vacancy_page_url: str) -> None:
        parsed_url = urlparse(vacancy_page_url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname not in {"hh.ru", "api.hh.ru"}
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.port not in {None, 443}
        ):
            raise ValueError("Expected a HeadHunter vacancy page URL")

    async def search_previews(
        self, query: VacancySearchQuery, vacancy_page_url: str, *, max_pages: int
    ) -> AsyncIterator[VacancyPreview]:
        parsed_url = urlparse(vacancy_page_url)
        self.validate_search_url(vacancy_page_url)
        for page in range(max_pages):
            params = parse_qs(parsed_url.query)
            params.update(text=query.text, page=page, per_page=100)
            if query.area_ids:
                params["area"] = query.area_ids
            if query.experience:
                params["experience"] = query.experience
            if query.published_after:
                params["date_from"] = query.published_after.isoformat()
            payload = await self._client.search_vacancies(params)
            for item in payload["items"]:
                salary = item.get("salary") or {}
                snippet = item.get("snippet") or {}
                yield VacancyPreview(
                    source=self.source_name,
                    external_id=item["id"],
                    title=item["name"],
                    url=item["alternate_url"],
                    company_name=(item.get("employer") or {}).get("name"),
                    location=(item.get("area") or {}).get("name"),
                    short_description="\n".join(
                        value for value in snippet.values() if isinstance(value, str)
                    ),
                    experience=(item.get("experience") or {}).get("id"),
                    published_at=item.get("published_at"),
                    salary_from=salary.get("from"),
                    salary_to=salary.get("to"),
                    salary_currency=salary.get("currency"),
                    salary_gross=salary.get("gross"),
                )
            if page + 1 >= payload["pages"]:
                break

    async def search(
        self,
        query: VacancySearchQuery,
    ) -> AsyncIterator[VacancyReference]:
        page = 0

        while True:
            params: dict[str, Any] = {
                "text": query.text,
                "page": page,
                "per_page": 100,
            }
            if query.area_ids:
                params["area"] = query.area_ids
            if query.experience:
                params["experience"] = query.experience
            if query.published_after is not None:
                params["date_from"] = query.published_after.isoformat()

            payload = await self._client.search_vacancies(
                params,
            )

            for item in payload["items"]:
                yield VacancyReference(
                    source=self.source_name,
                    external_id=item["id"],
                    title=item["name"],
                    url=item["alternate_url"],
                    published_at=item.get("published_at"),
                )

            page += 1

            if page >= payload["pages"]:
                break

    async def fetch_details(
        self,
        vacancy: VacancyReference,
    ) -> RawVacancy:
        payload = await self._client.get_vacancy(vacancy.external_id)

        return RawVacancy(
            source=self.source_name,
            external_id=vacancy.external_id,
            url=payload.get("alternate_url") or vacancy.url,
            title=payload.get("name"),
            raw_text=payload.get("description"),
            raw_payload=payload,
            published_at=payload.get("published_at") or vacancy.published_at,
            fetched_at=datetime.now(UTC),
        )
