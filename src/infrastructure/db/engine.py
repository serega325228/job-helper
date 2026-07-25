from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import DatabaseSettings

class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._engine = create_async_engine(
            settings.url,
            echo=settings.echo,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        self._configure_sqlite()

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self._engine.dispose()

    def _configure_sqlite(self) -> None:
        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragmas(
            dbapi_connection: object,
            connection_record: object,
        ) -> None:
            cursor = dbapi_connection.cursor()

            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")

            cursor.close()
