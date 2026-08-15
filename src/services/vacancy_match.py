from uuid import UUID

from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import VacancyEmbeddingSearchResult, VacancyMatchResult


class VacancyMatchService:
    def __init__(self, unit_of_work: SqlAlchemyUnitOfWork) -> None:
        self._uow = unit_of_work

    async def save_result(self, result: VacancyMatchResult) -> VacancyMatch:
        values = result.model_dump(mode="python")
        values["category"] = result.category.value

        async with self._uow as uow:
            vacancy_match = await uow.vacancy_matches.get_by_profile_and_vacancy(
                result.profile_id,
                result.vacancy_id,
            )
            if vacancy_match is None:
                vacancy_match = VacancyMatch(**values)
                uow.vacancy_matches.add(vacancy_match)
            else:
                for field, value in values.items():
                    setattr(vacancy_match, field, value)

            await uow.flush()

        return vacancy_match

    async def search_by_preferences(
        self,
        profile_id: UUID,
        hard_filters: VacancyHardFilters,
        *,
        limit: int = 100,
        title_weight: float = 0.4,
        candidate_limit: int = 100,
        per_preference_limit: int = 50,
    ) -> list[VacancyEmbeddingSearchResult]:
        async with self._uow as uow:
            return await uow.vacancy_matches.search_by_preferences(
                profile_id,
                hard_filters,
                limit=limit,
                title_weight=title_weight,
                candidate_limit=candidate_limit,
                per_preference_limit=per_preference_limit,
            )
