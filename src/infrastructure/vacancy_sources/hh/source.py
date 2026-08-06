from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.schemas.vacancy import RawVacancy, VacancyReference, VacancySearchQuery


class HhVacancySource:
    source_name = "hh"

    def __init__(self, client: HhApiClient) -> None:
        self._client = client

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
        payload = await self._client.get_vacancy(
            vacancy.external_id
        )

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
