from uuid import UUID

from sqlalchemy import and_, or_, select, true, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.repositories.vacancy_filters import build_vacancy_hard_filter_conditions
from src.schemas.vacancy import ProcessingStatus, VacancyHardFilters
from src.schemas.vacancy_match import VacancyEmbeddingSearchResult


class VacancyMatchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, vacancy_match: VacancyMatch) -> None:
        self._session.add(vacancy_match)

    def add_all(self, vacancy_matches: list[VacancyMatch]) -> None:
        self._session.add_all(vacancy_matches)

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

    async def get_by_profile_and_vacancy_ids(
        self,
        profile_id: UUID,
        vacancy_ids: list[UUID],
    ) -> list[VacancyMatch]:
        if not vacancy_ids:
            return []

        statement = select(VacancyMatch).where(
            VacancyMatch.profile_id == profile_id,
            VacancyMatch.vacancy_id.in_(vacancy_ids),
        )
        result = await self._session.scalars(statement)
        return list(result)

    async def list_for_profile(
        self,
        profile_id: UUID,
        *,
        limit: int = 100,
    ) -> list[VacancyMatch]:
        statement = (
            select(VacancyMatch)
            .where(
                VacancyMatch.profile_id == profile_id,
                VacancyMatch.processing_status == ProcessingStatus.COMPLETED,
                VacancyMatch.total_score.is_not(None),
            )
            .order_by(VacancyMatch.total_score.desc())
            .limit(limit)
        )
        result = await self._session.scalars(statement)
        return list(result)

    async def upsert_initial(self, values: list[dict]) -> None:
        if not values:
            return
        statement = insert(VacancyMatch).values(values)
        await self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[VacancyMatch.profile_id, VacancyMatch.vacancy_id],
                set_={
                    field: getattr(statement.excluded, field)
                    for field in values[0]
                    if field not in {"profile_id", "vacancy_id"}
                },
            )
        )

    async def fail_batch(self, batch_id: UUID) -> None:
        await self._session.execute(
            update(VacancyMatch)
            .where(
                VacancyMatch.batch_id == batch_id,
                VacancyMatch.processing_status.in_(
                    [
                        ProcessingStatus.PENDING_EMBEDDING,
                        ProcessingStatus.PENDING_RERANK,
                        ProcessingStatus.PENDING_LAYA,
                        ProcessingStatus.PENDING_SAVE,
                    ]
                ),
            )
            .values(processing_status=ProcessingStatus.FAILED)
        )

    async def get_batch_matches(
        self, batch_id: UUID, *, lock: bool = False
    ) -> list[VacancyMatch]:
        statement = (
            select(VacancyMatch)
            .where(VacancyMatch.batch_id == batch_id)
            .order_by(VacancyMatch.id)
        )
        if lock:
            statement = statement.with_for_update()
        return list(
            await self._session.scalars(
                statement.execution_options(populate_existing=True)
            )
        )

    async def embedding_scores(
        self,
        profile_id: UUID,
        hard_filters: VacancyHardFilters,
        vacancy_ids: list[UUID],
        preference_ids: list[UUID],
        title_weight: float,
        selection_context: dict | None = None,
    ) -> list[VacancyEmbeddingSearchResult]:
        # ponytail: pairwise scoring scales with cohort × preferences; use ANN for substantially larger cohorts.
        title = 1 - Vacancy.title_embedding.cosine_distance(
            PreferenceIntent.title_embedding
        )
        content = 1 - Vacancy.content_embedding.cosine_distance(
            PreferenceIntent.content_embedding
        )
        statement = (
            select(
                Vacancy.id.label("vacancy_id"),
                PreferenceIntent.id.label("preference_id"),
                title.label("title_similarity"),
                content.label("content_similarity"),
                (title * title_weight + content * (1 - title_weight)).label(
                    "combined_similarity"
                ),
            )
            .select_from(Vacancy)
            .join(PreferenceIntent, true())
            .where(
                Vacancy.id.in_(vacancy_ids),
                PreferenceIntent.id.in_(preference_ids),
                PreferenceIntent.profile_id == profile_id,
                PreferenceIntent.enabled.is_(True),
                *build_vacancy_hard_filter_conditions(hard_filters),
                Vacancy.title_embedding.is_not(None),
                Vacancy.content_embedding.is_not(None),
                PreferenceIntent.title_embedding.is_not(None),
                PreferenceIntent.content_embedding.is_not(None),
            )
            .order_by(Vacancy.id, PreferenceIntent.id)
        )
        if selection_context:
            statement = statement.where(
                or_(
                    *(
                        and_(
                            Vacancy.id.in_(
                                [UUID(value) for value in scope["vacancy_ids"]]
                            ),
                            PreferenceIntent.id.in_(
                                [UUID(value) for value in scope["preference_ids"]]
                            ),
                            *build_vacancy_hard_filter_conditions(
                                VacancyHardFilters.model_validate(scope["hard_filters"])
                            ),
                        )
                        for scope in selection_context.values()
                    )
                )
            )
        result = await self._session.execute(statement)
        return [
            VacancyEmbeddingSearchResult.model_validate(row)
            for row in result.mappings()
        ]

    async def get_amount_by_profile_id(self, profile_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(VacancyMatch)
            .where(
                VacancyMatch.profile_id == profile_id,
                VacancyMatch.processing_status == ProcessingStatus.COMPLETED,
                VacancyMatch.total_score.is_not(None),
            )
        )

        result = await self._session.scalar(stmt)
        return result if result else 0
