from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.company import Company
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.models.vacancy_preview import PreviewCollection, VacancyPreview
from src.schemas.vacancy import CollectionStatus, PreviewStatus
from src.schemas.vacancy import VacancyPreview as PreviewData


class VacancyPreviewRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve_sources(self, names: list[str]) -> list[Company]:
        builtins = {
            "hh": "https://api.hh.ru/vacancies",
            "linkedin": "https://www.linkedin.com/jobs/search/",
        }
        selected = set(builtins) if names == ["all"] else set(names) & builtins.keys()
        if selected:
            await self._session.execute(
                insert(Company)
                .values(
                    [
                        {"name": name, "careers_url": builtins[name]}
                        for name in sorted(selected)
                    ]
                )
                .on_conflict_do_nothing(index_elements=[Company.name]),
            )
        statement = select(Company).where(Company.enabled.is_(True))
        if names != ["all"]:
            statement = statement.where(Company.name.in_(names))
        sources = list(await self._session.scalars(statement.order_by(Company.id)))
        if names != ["all"] and set(names) != {source.name for source in sources}:
            raise ValueError("Unknown or disabled source name")
        return sources

    def add_collections(self, collections: list[PreviewCollection]) -> None:
        self._session.add_all(collections)

    async def get_collection(
        self, collection_id: UUID, *, lock: bool = False
    ) -> PreviewCollection | None:
        statement = select(PreviewCollection).where(
            PreviewCollection.id == collection_id
        )
        if lock:
            statement = statement.with_for_update(skip_locked=True, key_share=True)
        return await self._session.scalar(
            statement.execution_options(populate_existing=True)
        )

    async def get_by_ids(
        self, preview_ids: list[UUID], *, lock: bool = False
    ) -> list[VacancyPreview]:
        statement = (
            select(VacancyPreview)
            .where(VacancyPreview.id.in_(preview_ids))
            .order_by(VacancyPreview.id)
        )
        if lock:
            statement = statement.with_for_update(skip_locked=True)
        return list(
            await self._session.scalars(
                statement.execution_options(populate_existing=True)
            )
        )

    async def list_ready(
        self, profile_id: UUID, preview_ids: list[UUID] | None = None
    ) -> list[VacancyPreview]:
        statement = (
            select(VacancyPreview)
            .join(PreviewCollection)
            .where(
                PreviewCollection.profile_id == profile_id,
                VacancyPreview.status == PreviewStatus.READY,
            )
            .order_by(VacancyPreview.discovered_at.desc(), VacancyPreview.id)
            .with_for_update(of=VacancyPreview, skip_locked=True)
        )
        if preview_ids is not None:
            statement = statement.where(VacancyPreview.id.in_(preview_ids))
        return list(await self._session.scalars(statement))

    async def list_for_profile(
        self, profile_id: UUID, limit: int
    ) -> list[VacancyPreview]:
        statement = (
            select(VacancyPreview)
            .join(PreviewCollection)
            .where(
                PreviewCollection.profile_id == profile_id,
            )
            .order_by(VacancyPreview.discovered_at.desc(), VacancyPreview.id)
            .limit(limit)
        )
        return list(await self._session.scalars(statement))

    async def save_new(
        self,
        collection_id: UUID,
        previews: list[PreviewData],
        *,
        description_min_length: int,
    ) -> list[VacancyPreview]:
        if not previews:
            return []
        normalized = {
            (
                " ".join((preview.company_name or "").lower().split()),
                " ".join(preview.title.lower().split()),
            )
            for preview in previews
            if preview.company_name
        }
        keys = {(preview.source, preview.external_id) for preview in previews}
        locks = {f"preview:{source}:{external_id}" for source, external_id in keys}
        locks.update(f"repost:{company}:{title}" for company, title in normalized)
        for key in sorted(locks):
            await self._session.execute(
                select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0)))
            )
        existing_keys: set[tuple[str, str]] = set()
        existing_reposts: set[tuple[str, str]] = set()
        now = datetime.now(UTC)
        for model in (VacancyPreview, Vacancy):
            statement = select(
                model.source,
                model.external_id,
                model.normalized_company,
                model.normalized_title,
                model.discovered_at,
            ).where(
                or_(
                    tuple_(model.source, model.external_id).in_(keys),
                    (model.discovered_at >= now - timedelta(days=7))
                    & (model.discovered_at <= now)
                    & tuple_(model.normalized_company, model.normalized_title).in_(
                        normalized
                    ),
                ),
            )
            for (
                source,
                external_id,
                company,
                title,
                discovered_at,
            ) in await self._session.execute(statement):
                existing_keys.add((source, external_id))
                if company and now - timedelta(days=7) <= discovered_at <= now:
                    existing_reposts.add((company, title))
        values = []
        for preview in previews:
            key = (preview.source, preview.external_id)
            repost = (
                " ".join((preview.company_name or "").lower().split()),
                " ".join(preview.title.lower().split()),
            )
            if key in existing_keys or (repost[0] and repost in existing_reposts):
                continue
            existing_keys.add(key)
            if repost[0]:
                existing_reposts.add(repost)
            # ponytail: description length approximates preview sufficiency; replace with a labeled relevance gate when calibrated.
            values.append(
                {
                    **preview.model_dump(mode="python"),
                    "url": str(preview.url),
                    "collection_id": collection_id,
                    "status": PreviewStatus.PENDING_LAYA
                    if len((preview.short_description or "").strip())
                    >= description_min_length
                    else PreviewStatus.PENDING_FILTER,
                }
            )
        if not values:
            return []
        statement = (
            insert(VacancyPreview)
            .values(values)
            .on_conflict_do_nothing(
                index_elements=[VacancyPreview.source, VacancyPreview.external_id],
            )
            .returning(VacancyPreview)
        )
        return list(await self._session.scalars(statement))

    async def stale_collections(
        self, cutoff: datetime, limit: int
    ) -> list[PreviewCollection]:
        return list(
            await self._session.scalars(
                select(PreviewCollection)
                .where(
                    PreviewCollection.status == CollectionStatus.PENDING,
                    PreviewCollection.updated_at < cutoff,
                )
                .order_by(PreviewCollection.updated_at)
                .limit(limit)
                .with_for_update(skip_locked=True),
            )
        )

    async def stale_previews(
        self, cutoff: datetime, limit: int
    ) -> list[VacancyPreview]:
        return list(
            await self._session.scalars(
                select(VacancyPreview)
                .where(
                    VacancyPreview.status.in_(
                        [PreviewStatus.PENDING_FILTER, PreviewStatus.PENDING_LAYA]
                    ),
                    VacancyPreview.updated_at < cutoff,
                )
                .order_by(VacancyPreview.updated_at)
                .limit(limit)
                .with_for_update(skip_locked=True),
            )
        )
