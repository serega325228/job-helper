from types import TracebackType
from typing import Self
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from repositories.resume import ResumeRepository
from src.repositories.profile import ProfileRepository
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_match import VacancyMatchRepository


class SqlAlchemyUnitOfWork:
    def __init__(
        self,
        session: AsyncSession,
        profile_repository: ProfileRepository,
        vacancy_repository: VacancyRepository,
        vacancy_match_repository: VacancyMatchRepository,
        resume_repository: ResumeRepository,
    ) -> None:
        self._session = session

        self.profiles = profile_repository
        self.vacancies = vacancy_repository
        self.vacancy_matches = vacancy_match_repository
        self.resumes = resume_repository
        self._active = False

    async def __aenter__(self) -> Self:
        if self._active:
            raise RuntimeError("Unit of Work is already active")

        await self._session.begin()
        self._active = True

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            self._active = False

    async def flush(self) -> None:
        await self._session.flush()

    async def get_stats(
        self,
        profile_id: UUID
    ) -> dict:
        async with self:
             resumes = await self.resumes.get_amount_by_profile_id(profile_id)
             scrapped_vacancies = await self.vacancies.get_amount()
             matched_vacancies = self.vacancy_matches.get_amount_by_profile_id(profile_id)

        return {
            "resumes": resumes,
            "scrapped_vacancies": scrapped_vacancies,
            "matched_vacancies": matched_vacancies
        }
