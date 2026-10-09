from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile


class ProfileRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, profile: Profile) -> Profile:
        self._session.add(profile)
        await self._session.flush()
        return profile

    async def get_by_id(self, id: UUID, *, lock: bool = False) -> Profile | None:
        statement = select(Profile).where(Profile.id == id)
        if lock:
            statement = statement.with_for_update()
        return await self._session.scalar(
            statement.execution_options(populate_existing=True)
        )

    async def get_preferences_by_profile_id(
        self,
        profile_id: UUID,
        *,
        enabled_only: bool = False,
    ) -> list[PreferenceIntent]:
        stmt = (
            select(PreferenceIntent)
            .where(PreferenceIntent.profile_id == profile_id)
            .order_by(PreferenceIntent.created_at, PreferenceIntent.id)
        )
        if enabled_only:
            stmt = stmt.where(PreferenceIntent.enabled.is_(True))
        result = await self._session.scalars(
            stmt.execution_options(populate_existing=True)
        )
        return list(result)

    async def get_preferences_by_ids(
        self,
        profile_id: UUID,
        preference_ids: list[UUID],
        *,
        enabled_only: bool = False,
    ) -> list[PreferenceIntent]:
        stmt = select(PreferenceIntent).where(
            PreferenceIntent.id.in_(preference_ids),
            PreferenceIntent.profile_id == profile_id,
        )
        if enabled_only:
            stmt = stmt.where(PreferenceIntent.enabled.is_(True))
        result = await self._session.scalars(stmt)
        return list(result)
