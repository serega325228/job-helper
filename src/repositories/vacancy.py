from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.vacancy import Vacancy


class VacancyRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, vacancy: Vacancy) -> Vacancy:
        self._session.add(vacancy)
        await self._session.flush()
        return vacancy

    def add_all(self, vacancies: list[Vacancy]) -> None:
        self._session.add_all(vacancies)

    async def get_by_id(self, vacancy_id: UUID) -> Vacancy | None:
        return await self._session.get(Vacancy, vacancy_id)

    async def get_by_external_keys(
        self,
        keys: set[tuple[str, str]],
    ) -> list[Vacancy]:
        if not keys:
            return []

        statement = select(Vacancy).where(
            tuple_(Vacancy.source, Vacancy.external_id).in_(keys),
        )
        result = await self._session.scalars(statement)
        return list(result)
