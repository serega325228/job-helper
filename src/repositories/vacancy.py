from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src.config.settings import Settings
from src.infrastructure.models.vacancy import Vacancy, VacancyBatch
from src.repositories.vacancy_filters import build_vacancy_hard_filter_conditions
from src.schemas.vacancy import BatchStatus, ProcessingStatus, VacancyHardFilters


class VacancyRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, vacancy: Vacancy) -> Vacancy:
        self._session.add(vacancy)
        await self._session.flush()
        return vacancy

    def add_all(self, vacancies: list[Vacancy]) -> None:
        self._session.add_all(vacancies)

    def add_batch(self, batch: VacancyBatch) -> None:
        self._session.add(batch)

    async def create_from_previews(self, values: list[dict]) -> list[Vacancy]:
        if not values:
            return []
        statement = (
            insert(Vacancy)
            .values(values)
            .on_conflict_do_nothing(
                index_elements=[Vacancy.source, Vacancy.external_id],
            )
            .returning(Vacancy)
        )
        await self._session.execute(statement)
        return await self.get_by_external_keys(
            {(value["source"], value["external_id"]) for value in values}
        )

    async def get_batch(self, batch_id: UUID) -> VacancyBatch | None:
        statement = (
            select(VacancyBatch).where(VacancyBatch.id == batch_id).with_for_update()
        )
        return await self._session.scalar(
            statement.execution_options(populate_existing=True)
        )

    async def get_stage(
        self, vacancy_ids: list[UUID], status: ProcessingStatus
    ) -> list[Vacancy]:
        statement = (
            select(Vacancy)
            .where(
                Vacancy.id.in_(vacancy_ids),
                Vacancy.processing_status == status,
            )
            .order_by(Vacancy.id)
            .with_for_update()
        )
        return list(
            await self._session.scalars(
                statement.execution_options(populate_existing=True)
            )
        )

    async def busy_vacancy_ids(self, profile_id: UUID) -> set[UUID]:
        batches = await self._session.scalars(
            select(VacancyBatch).where(
                VacancyBatch.profile_id == profile_id,
                VacancyBatch.status.not_in([BatchStatus.COMPLETED, BatchStatus.FAILED]),
            )
        )
        return {UUID(value) for batch in batches for value in batch.vacancy_ids}

    async def ready_batch_ids(self, vacancy_ids: list[UUID]) -> set[UUID]:
        if not vacancy_ids:
            return set()
        statement = select(VacancyBatch).where(
            VacancyBatch.status == BatchStatus.DETAILS,
            or_(
                *(
                    VacancyBatch.vacancy_ids.contains([str(identifier)])
                    for identifier in vacancy_ids
                )
            ),
        )
        ready = set()
        for batch in await self._session.scalars(statement):
            vacancies = await self.get_vacancies_by_ids(
                [UUID(value) for value in batch.vacancy_ids]
            )
            if not any(
                vacancy.processing_status
                in {ProcessingStatus.PENDING_SCRAPE, ProcessingStatus.PENDING_PARSE}
                for vacancy in vacancies
            ):
                ready.add(batch.id)
        return ready

    async def claim_details(
        self,
        vacancy_ids: list[UUID],
        status: ProcessingStatus,
        token: UUID,
        settings: Settings,
    ) -> list[Vacancy]:
        rows = await self.get_stage(vacancy_ids, status)
        claimed = []
        cutoff = datetime.now(UTC) - timedelta(
            seconds=settings.taskiq.lease_timeout_seconds
        )
        for row in rows:
            if row.processing_token is not None and row.updated_at >= cutoff:
                continue
            if row.attempts >= settings.taskiq.stage_attempts:
                row.processing_status = ProcessingStatus.FAILED
                row.error = "Stage attempts exhausted"
                row.processing_token = None
                continue
            row.processing_token = token
            row.attempts += 1
            row.updated_at = datetime.now(UTC)
            claimed.append(row)
        return claimed

    async def finish_details(self, values: dict[UUID, dict], token: UUID) -> None:
        for row in await self.get_vacancies_by_ids(list(values), lock=True):
            if row.processing_token != token:
                continue
            for field, value in values[row.id].items():
                setattr(row, field, value)
            row.processing_token = None
            row.attempts = 0
            row.error = None

    async def fail_details(
        self, vacancy_ids: list[UUID], token: UUID, error: Exception, settings: Settings
    ):
        for row in await self.get_vacancies_by_ids(vacancy_ids, lock=True):
            if row.processing_token == token:
                row.processing_token = None
                row.error = str(error)
                if (
                    isinstance(error, ValueError)
                    or row.attempts >= settings.taskiq.stage_attempts
                ):
                    row.processing_status = ProcessingStatus.FAILED

    async def stale_vacancies(
        self, cutoff: datetime, lease_cutoff: datetime, limit: int
    ) -> list[Vacancy]:
        statement = (
            select(Vacancy)
            .where(
                Vacancy.processing_status.in_(
                    [
                        ProcessingStatus.PENDING_SCRAPE,
                        ProcessingStatus.PENDING_PARSE,
                    ]
                ),
                Vacancy.updated_at < cutoff,
                or_(
                    Vacancy.processing_token.is_(None),
                    Vacancy.updated_at < lease_cutoff,
                ),
            )
            .order_by(Vacancy.updated_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list(await self._session.scalars(statement))

    async def stale_batches(
        self, cutoff: datetime, lease_cutoff: datetime, limit: int
    ) -> list[VacancyBatch]:
        statement = (
            select(VacancyBatch)
            .where(
                VacancyBatch.status.not_in([BatchStatus.COMPLETED, BatchStatus.FAILED]),
                VacancyBatch.updated_at < cutoff,
                or_(
                    VacancyBatch.processing_token.is_(None),
                    VacancyBatch.updated_at < lease_cutoff,
                ),
            )
            .order_by(VacancyBatch.updated_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list(await self._session.scalars(statement))

    async def get_by_id(self, vacancy_id: UUID) -> Vacancy | None:
        return await self._session.get(Vacancy, vacancy_id)

    async def get_by_external_keys(
        self,
        keys: set[tuple[str, str]],
        *,
        urls: set[str] | None = None,
    ) -> list[Vacancy]:
        if not keys and not urls:
            return []

        # ponytail: URL fallback scans unindexed URLs; add a URL index as storage grows.
        stmt = select(Vacancy).where(
            or_(
                tuple_(Vacancy.source, Vacancy.external_id).in_(keys),
                Vacancy.url.in_(urls or set()),
            ),
        )
        result = await self._session.scalars(stmt)
        return list(result)

    async def get_vacancies_by_ids(
        self,
        vacancy_ids: list[UUID],
        *,
        lock: bool = False,
    ) -> list[Vacancy]:
        if not vacancy_ids:
            return []

        stmt = select(Vacancy).where(Vacancy.id.in_(vacancy_ids))
        if lock:
            stmt = stmt.order_by(Vacancy.id).with_for_update()
        result = await self._session.scalars(
            stmt.execution_options(populate_existing=True)
        )
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
        stmt = select(func.count()).select_from(Vacancy)
        result = await self._session.scalar(stmt)
        return result if result else 0
