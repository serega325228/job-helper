from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.profile import ProfileRepository


class SqlAlchemyUnitOfWork:
    def __init__(
        self,
        session: AsyncSession,
        profile_repository: ProfileRepository,
    ) -> None:
        self._session = session

        self.profiles = profile_repository
        self._active = False

    async def __aenter__(self) -> Self:
        if self._active:
            raise RuntimeError("Unit of Work is already active")

        await self._session.begin()
        self._active = True

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            self._active = False

    async def flush(self) -> None:
        await self._session.flush()
