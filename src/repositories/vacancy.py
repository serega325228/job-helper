from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src.infrastructure.models.vacancy import Vacancy
from src.repositories.vacancy_filters import build_vacancy_hard_filter_conditions
from src.schemas.vacancy import VacancyHardFilters


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

        stmt = select(Vacancy).where(
            tuple_(Vacancy.source, Vacancy.external_id).in_(keys),
        )
        result = await self._session.scalars(stmt)
        return list(result)

    async def get_vacancies_by_ids(
        self,
        vacancy_ids: list[UUID],
    ) -> list[Vacancy]:
        if not vacancy_ids:
            return []

        stmt = select(Vacancy).where(Vacancy.id.in_(vacancy_ids))
        result = await self._session.scalars(stmt)
        return list(result)

    async def list_by_hard_filters(
        self,
        filters: VacancyHardFilters,
        *,
        limit: int = 100,
    ) -> list[Vacancy]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        stmt = (
            select(Vacancy)
            .where(*build_vacancy_hard_filter_conditions(filters))
            .order_by(
                Vacancy.published_at.desc().nulls_last(),
                Vacancy.discovered_at.desc(),
                Vacancy.id,
            )
            .limit(limit)
        )
        result = await self._session.scalars(stmt)
        return list(result)

    async def ids_by_hard_filters(
        self,
        filters: VacancyHardFilters,
        *,
        limit: int = 100,
    ) -> list[UUID]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        stmt = (
            select(Vacancy.id)
            .where(*build_vacancy_hard_filter_conditions(filters))
            .order_by(
                Vacancy.published_at.desc().nulls_last(),
                Vacancy.discovered_at.desc(),
                Vacancy.id,
            )
            .limit(limit)
        )
        result = await self._session.scalars(stmt)
        return list(result)

    async def get_amount(self) -> int:
        stmt = (
            select(func.count())
            .select_from(Vacancy)
        )
        result = await self._session.scalar(stmt)
        return result if result else 0
