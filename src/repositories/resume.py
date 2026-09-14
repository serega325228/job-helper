from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.models.resume import Resume


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
