from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.vacancy_match import VacancyMatch


class VacancyMatchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, vacancy_match: VacancyMatch) -> None:
        self._session.add(vacancy_match)

    async def get_by_id(self, match_id: UUID) -> VacancyMatch | None:
        return await self._session.get(VacancyMatch, match_id)

    async def get_by_profile_and_vacancy(
        self,
        profile_id: UUID,
        vacancy_id: UUID,
    ) -> VacancyMatch | None:
        statement = select(VacancyMatch).where(
            VacancyMatch.profile_id == profile_id,
            VacancyMatch.vacancy_id == vacancy_id,
        )
        return await self._session.scalar(statement)

    async def list_for_profile(
        self,
        profile_id: UUID,
        *,
        limit: int = 100,
    ) -> list[VacancyMatch]:
        statement = (
            select(VacancyMatch)
            .where(VacancyMatch.profile_id == profile_id)
            .order_by(VacancyMatch.total_score.desc())
            .limit(limit)
        )
        result = await self._session.scalars(statement)
        return list(result)
