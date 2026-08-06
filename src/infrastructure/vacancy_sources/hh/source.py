from collections.abc import AsyncIterator
from datetime import datetime, UTC

from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.infrastructure.vacancy_sources.hh.schemas import RawVacancy, VacancyReference, VacancySearchQuery


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
            payload = await self._client.search_vacancies(
                {
                    "text": query.text,
                    "area": query.area_ids,
                    "page": page,
                    "per_page": 100,
                }
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
            url=vacancy.url,
            title=payload.get("name"),
            raw_text=payload.get("description"),
            raw_payload=payload,
            fetched_at=datetime.now(UTC),
        )
