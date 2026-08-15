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

    async def get_by_id(self, id: UUID) -> Profile | None:
        return await self._session.get(Profile, id)

    async def get_preferences_by_profile_id(
        self,
        profile_id: UUID,
    ) -> list[PreferenceIntent]:
        stmt = (
            select(PreferenceIntent)
            .where(PreferenceIntent.profile_id == profile_id)
            .order_by(PreferenceIntent.created_at, PreferenceIntent.id)
        )
        result = await self._session.scalars(stmt)
        return list(result)

    async def get_preferences_by_ids(
        self,
        profile_id: UUID,
        preference_ids: list[UUID],
    ) -> list[PreferenceIntent]:
        stmt = (
            select(PreferenceIntent)
            .where(
                PreferenceIntent.id.in_(preference_ids),
                PreferenceIntent.profile_id == profile_id
            )
        )
        result = await self._session.scalars(stmt)
        return list(result)
