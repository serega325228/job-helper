from uuid import UUID

from sqlalchemy import func, or_, select, tuple_, union
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.vacancy import Vacancy
from src.schemas.vacancy import (
    VacancyEmbeddingSearchResult,
    VacancyHardFilters,
)


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

        conditions = []
        if filters.sources:
            conditions.append(Vacancy.source.in_(filters.sources))
        if filters.statuses:
            conditions.append(Vacancy.status.in_(filters.statuses))
        if filters.area_ids:
            conditions.append(Vacancy.area_id.in_(filters.area_ids))
        if filters.countries:
            conditions.append(Vacancy.country.in_(filters.countries))
        if filters.cities:
            conditions.append(Vacancy.city.in_(filters.cities))
        if filters.company_names:
            conditions.append(Vacancy.company_name.in_(filters.company_names))
        if filters.excluded_company_names:
            conditions.append(
                or_(
                    Vacancy.company_name.is_(None),
                    Vacancy.company_name.not_in(filters.excluded_company_names),
                ),
            )
        if filters.work_formats:
            conditions.append(Vacancy.work_format.in_(filters.work_formats))
        if filters.employment_types:
            conditions.append(
                Vacancy.employment_type.in_(filters.employment_types),
            )
        if filters.work_schedules:
            conditions.append(Vacancy.work_schedule.in_(filters.work_schedules))
        if filters.experience:
            conditions.append(Vacancy.experience.in_(filters.experience))
        if filters.seniorities:
            conditions.append(Vacancy.seniority.in_(filters.seniorities))
        if filters.salary_min is not None:
            conditions.append(
                func.coalesce(Vacancy.salary_to, Vacancy.salary_from)
                >= filters.salary_min,
            )
        if filters.salary_currency is not None:
            conditions.append(Vacancy.salary_currency == filters.salary_currency)
        if filters.salary_gross is not None:
            conditions.append(Vacancy.salary_gross == filters.salary_gross)
        if filters.published_after is not None:
            conditions.append(Vacancy.published_at >= filters.published_after)

        stmt = (
            select(Vacancy)
            .where(*conditions)
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

        conditions = []
        if filters.sources:
            conditions.append(Vacancy.source.in_(filters.sources))
        if filters.statuses:
            conditions.append(Vacancy.status.in_(filters.statuses))
        if filters.area_ids:
            conditions.append(Vacancy.area_id.in_(filters.area_ids))
        if filters.countries:
            conditions.append(Vacancy.country.in_(filters.countries))
        if filters.cities:
            conditions.append(Vacancy.city.in_(filters.cities))
        if filters.company_names:
            conditions.append(Vacancy.company_name.in_(filters.company_names))
        if filters.excluded_company_names:
            conditions.append(
                or_(
                    Vacancy.company_name.is_(None),
                    Vacancy.company_name.not_in(filters.excluded_company_names),
                ),
            )
        if filters.work_formats:
            conditions.append(Vacancy.work_format.in_(filters.work_formats))
        if filters.employment_types:
            conditions.append(
                Vacancy.employment_type.in_(filters.employment_types),
            )
        if filters.work_schedules:
            conditions.append(Vacancy.work_schedule.in_(filters.work_schedules))
        if filters.experience:
            conditions.append(Vacancy.experience.in_(filters.experience))
        if filters.seniorities:
            conditions.append(Vacancy.seniority.in_(filters.seniorities))
        if filters.salary_min is not None:
            conditions.append(
                func.coalesce(Vacancy.salary_to, Vacancy.salary_from)
                >= filters.salary_min,
            )
        if filters.salary_currency is not None:
            conditions.append(Vacancy.salary_currency == filters.salary_currency)
        if filters.salary_gross is not None:
            conditions.append(Vacancy.salary_gross == filters.salary_gross)
        if filters.published_after is not None:
            conditions.append(Vacancy.published_at >= filters.published_after)

        stmt = (
            select(Vacancy.id)
            .where(*conditions)
            .order_by(
                Vacancy.published_at.desc().nulls_last(),
                Vacancy.discovered_at.desc(),
                Vacancy.id,
            )
            .limit(limit)
        )
        result = await self._session.scalars(stmt)
        return list(result)
