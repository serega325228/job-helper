from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.vacancy import Vacancy


class VacancyRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, vacancy: Vacancy) -> Vacancy:
        self._session.add(vacancy)
        await self._session.flush()
        return vacancy

    async def get_by_id(self, id: UUID) -> Vacancy | None:
        return await self._session.get(Vacancy, id)
