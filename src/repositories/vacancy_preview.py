from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.company import Company
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.models.vacancy_preview import (
    CollectedPreview,
    PreviewCollection,
    VacancyPreview,
)
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

    async def get_collection(self, collection_id: UUID, *, lock: bool = False):
        statement = select(PreviewCollection).where(
            PreviewCollection.id == collection_id
        )
        if lock:
            statement = statement.with_for_update()
        return await self._session.scalar(
            statement.execution_options(populate_existing=True)
        )

    async def get_by_ids(self, preview_ids: list[UUID]) -> list[VacancyPreview]:
        return list(
            await self._session.scalars(
                select(VacancyPreview)
                .where(VacancyPreview.id.in_(preview_ids))
                .execution_options(populate_existing=True)
            )
        )

    async def get_collected(
        self,
        *,
        ids: list[UUID] | None = None,
        profile_id: UUID | None = None,
        collection_id: UUID | None = None,
        status: PreviewStatus | None = None,
        limit: int | None = None,
        lock: bool = False,
    ) -> list:
        statement = (
            select(CollectedPreview, VacancyPreview)
            .join(VacancyPreview, VacancyPreview.id == CollectedPreview.preview_id)
            .join(
                PreviewCollection,
                PreviewCollection.id == CollectedPreview.collection_id,
            )
        )
        if ids is not None:
            statement = statement.where(CollectedPreview.id.in_(ids))
        if profile_id is not None:
            statement = statement.where(PreviewCollection.profile_id == profile_id)
        if collection_id is not None:
            statement = statement.where(CollectedPreview.collection_id == collection_id)
        if status is not None:
            statement = statement.where(CollectedPreview.status == status)
        statement = statement.order_by(
            VacancyPreview.discovered_at.desc(), CollectedPreview.id
        )
        if limit is not None:
            statement = statement.limit(limit)
        if lock:
            statement = statement.with_for_update(of=CollectedPreview)
        return list(
            await self._session.execute(
                statement.execution_options(populate_existing=True)
            )
        )

    async def save_new(
        self, collection_id: UUID, previews: list[PreviewData]
    ) -> list[UUID]:
        if not previews:
            return []
        # ponytail: company/title can merge distinct openings; add location/conditions if false positives grow.
        normalized = {
            (
                " ".join((preview.company_name or "").casefold().split()),
                " ".join(preview.title.casefold().split()),
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
        now = datetime.now(UTC)
        exact, reposts = {}, {}
        for model in (VacancyPreview, Vacancy):
            statement = (
                select(model)
                .where(
                    or_(
                        tuple_(model.source, model.external_id).in_(keys),
                        (model.discovered_at >= now - timedelta(days=7))
                        & (model.discovered_at <= now)
                        & tuple_(model.normalized_company, model.normalized_title).in_(
                            normalized
                        ),
                    )
                )
                .order_by(model.discovered_at.desc())
            )
            for row in await self._session.scalars(statement):
                exact.setdefault((row.source, row.external_id), row)
                if (
                    row.normalized_company
                    and now - timedelta(days=7) <= row.discovered_at <= now
                ):
                    reposts.setdefault(
                        (row.normalized_company, row.normalized_title), row
                    )
        new_values, preview_ids = [], set()
        for data in previews:
            key = (data.source, data.external_id)
            repost = (
                " ".join((data.company_name or "").casefold().split()),
                " ".join(data.title.casefold().split()),
            )
            existing = exact.get(key) or (reposts.get(repost) if repost[0] else None)
            if isinstance(existing, Vacancy):
                if existing.preview_id:
                    preview_ids.add(existing.preview_id)
                    continue
                data = PreviewData.model_validate(existing)
                key = (data.source, data.external_id)
            if isinstance(existing, VacancyPreview):
                preview_ids.add(existing.id)
                continue
            identifier = uuid4()
            values = {
                **data.model_dump(mode="python"),
                "id": identifier,
                "url": str(data.url),
                "normalized_company": repost[0] or None,
                "normalized_title": repost[1],
            }
            new_values.append(values)
            row = VacancyPreview(**values)
            exact[key] = row
            if repost[0]:
                reposts[repost] = row
            preview_ids.add(identifier)
        if new_values:
            await self._session.execute(insert(VacancyPreview).values(new_values))
        await self._session.execute(
            insert(CollectedPreview)
            .values(
                [
                    {"collection_id": collection_id, "preview_id": identifier}
                    for identifier in sorted(preview_ids)
                ]
            )
            .on_conflict_do_nothing(
                index_elements=[
                    CollectedPreview.collection_id,
                    CollectedPreview.preview_id,
                ]
            )
        )
        return [
            collected.id
            for collected, _ in await self.get_collected(
                collection_id=collection_id, status=PreviewStatus.PENDING_FILTER
            )
            if collected.preview_id in preview_ids
        ]

    async def stale_collections(
        self, cutoff: datetime, lease_cutoff: datetime, limit: int
    ):
        return list(
            await self._session.scalars(
                select(PreviewCollection)
                .where(
                    PreviewCollection.status == CollectionStatus.PENDING,
                    PreviewCollection.updated_at < cutoff,
                    or_(
                        PreviewCollection.processing_token.is_(None),
                        PreviewCollection.updated_at < lease_cutoff,
                    ),
                )
                .order_by(PreviewCollection.updated_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )

    async def stale_previews(self, cutoff: datetime, limit: int):
        return list(
            await self._session.scalars(
                select(CollectedPreview)
                .where(
                    CollectedPreview.status == PreviewStatus.PENDING_FILTER,
                    CollectedPreview.updated_at < cutoff,
                )
                .order_by(CollectedPreview.updated_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
