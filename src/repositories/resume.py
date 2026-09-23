
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.models.resume import Resume
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.functions import func


class ResumeRepository:
    def __init__(
        self,
        session: AsyncSession,
    ):
        self._session = session

    async def create(self, resume: Resume) -> Resume:
        self._session.add(resume)
        await self._session.flush()
        return resume

    async def get_amount_by_profile_id(self, profile_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(Resume)
            .where(Resume.profile_id == profile_id)
        )
        result = await self._session.scalar(stmt)
        return result if result else 0
