from uuid import UUID

from sqlalchemy import select, true, union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.repositories.vacancy_filters import build_vacancy_hard_filter_conditions
from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import VacancyEmbeddingSearchResult


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
        if not 0 <= title_weight <= 1:
            raise ValueError("title_weight must be between zero and one")
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be greater than zero")
        if per_preference_limit < 1:
            raise ValueError("per_preference_limit must be greater than zero")

        title_distance = Vacancy.title_embedding.cosine_distance(
            PreferenceIntent.title_embedding,
        )
        content_distance = Vacancy.content_embedding.cosine_distance(
            PreferenceIntent.content_embedding,
        )
        content_weight = 1 - title_weight
        combined_distance = (
            title_distance * title_weight + content_distance * content_weight
        )
        vacancy_conditions = [
            *build_vacancy_hard_filter_conditions(hard_filters),
            Vacancy.title_embedding.is_not(None),
            Vacancy.content_embedding.is_not(None),
        ]

        title_candidates = (
            select(Vacancy.id.label("vacancy_id"))
            .where(*vacancy_conditions)
            .order_by(title_distance)
            .limit(candidate_limit)
            .correlate(PreferenceIntent)
        )
        content_candidates = (
            select(Vacancy.id.label("vacancy_id"))
            .where(*vacancy_conditions)
            .order_by(content_distance)
            .limit(candidate_limit)
            .correlate(PreferenceIntent)
        )
        candidates = union(title_candidates, content_candidates).subquery(
            "candidates",
        )

        nearest_neighbors = (
            select(
                Vacancy.id.label("vacancy_id"),
                (1 - title_distance).label("title_similarity"),
                (1 - content_distance).label("content_similarity"),
                combined_distance.label("distance"),
            )
            .join(candidates, Vacancy.id == candidates.c.vacancy_id)
            .order_by(combined_distance)
            .limit(per_preference_limit)
            .correlate(PreferenceIntent)
            .lateral("nn")
        )
        semantic_candidates = (
            select(
                PreferenceIntent.id.label("preference_id"),
                nearest_neighbors.c.vacancy_id,
                nearest_neighbors.c.title_similarity,
                nearest_neighbors.c.content_similarity,
                nearest_neighbors.c.distance,
            )
            .join(nearest_neighbors, true())
            .where(
                PreferenceIntent.profile_id == profile_id,
                PreferenceIntent.enabled.is_(True),
                PreferenceIntent.title_embedding.is_not(None),
                PreferenceIntent.content_embedding.is_not(None),
            )
            .cte("semantic_candidates")
        )
        rank = (
            func.row_number()
            .over(
                partition_by=semantic_candidates.c.vacancy_id,
                order_by=semantic_candidates.c.distance,
            )
            .label("rank")
        )
        best_match = select(semantic_candidates, rank).cte("best_match")
        statement = (
            select(
                best_match.c.vacancy_id,
                best_match.c.preference_id,
                best_match.c.title_similarity,
                best_match.c.content_similarity,
                (1 - best_match.c.distance).label("combined_similarity"),
            )
            .where(best_match.c.rank == 1)
            .order_by(best_match.c.distance)
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return [
            VacancyEmbeddingSearchResult.model_validate(row)
            for row in result.mappings()
        ]
