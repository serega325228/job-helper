from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from infrastructure.models.preference_intent import PreferenceIntent
from infrastructure.models.vacancy import Vacancy
from schemas.vacancy_match import VacancyEmbeddingSearchResult
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

    #fix distance/similarity and add hard filters to this query
    async def search_by_preferences(
            self,
            profile_id: UUID,
            vacancy_ids: list[UUID],
            limit: int = 100,
            title_weight: float = 0.4,
        ) -> list[VacancyEmbeddingSearchResult]:
            if not 0 <= title_weight <= 1:
                raise ValueError("title_weight must be between zero and one")
            if limit < 1:
                raise ValueError("limit must be greater than zero")
            if vacancy_ids == []:
                return []

            title_distance = (
                Vacancy.title_embedding.cosine_distance(
                    PreferenceIntent.title_embedding
                )
            )

            content_distance = (
                Vacancy.content_embedding.cosine_distance(
                    PreferenceIntent.content_embedding
                )
            )

            content_weight = 1 - title_weight
            combined_distance = (
                title_distance * title_weight
                + content_distance * content_weight
            )

            rank = func.row_number().over(
                partition_by=Vacancy.id,
                order_by=combined_distance.asc(),
            ).label("rank")

            scored = (
                select(
                    Vacancy.id.label("vacancy_id"),
                    PreferenceIntent.id.label("preference_id"),

                    title_distance.label("title_distance"),
                    content_distance.label("content_distance"),
                    combined_distance.label("combined_distance"),

                    rank,
                )
                .join(PreferenceIntent, true())
                .where(
                    Vacancy.id.in_(vacancy_ids),

                    PreferenceIntent.profile_id == profile_id,
                    PreferenceIntent.enabled.is_(True),

                    Vacancy.title_embedding.is_not(None),
                    Vacancy.content_embedding.is_not(None),
                    PreferenceIntent.title_embedding.is_not(None),
                    PreferenceIntent.content_embedding.is_not(None),
                )
                .cte("scored")
            )

            stmt = (
                select(
                    scored.c.vacancy_id,
                    scored.c.preference_id,
                    scored.c.title_distance,
                    scored.c.content_distance,
                    scored.c.combined_distance,
                )
                .where(scored.c.rank == 1)
                .order_by(scored.c.combined_distance.asc())
                .limit(limit)
            )

            result = await self._session.execute(stmt)
            return [
                VacancyEmbeddingSearchResult.model_validate(row)
                for row in result.mappings()
            ]
