from collections.abc import AsyncIterator
from typing import Protocol

from src.schemas.vacancy import RawVacancy, VacancyReference, VacancySearchQuery


class VacancySource(Protocol):
    source_name: str

    def search(
        self,
        query: VacancySearchQuery,
    ) -> AsyncIterator[VacancyReference]: ...

    async def fetch_details(
        self,
        vacancy: VacancyReference,
    ) -> RawVacancy: ...
