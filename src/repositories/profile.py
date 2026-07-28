from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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
