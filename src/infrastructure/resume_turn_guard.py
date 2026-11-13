from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import blake2b
from uuid import UUID


class ResumeTurnGuard:
    """Cross-process turn serialization; separate connection avoids saver-pool starvation."""

    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo

    @asynccontextmanager
    async def hold(self, profile_id: UUID) -> AsyncIterator[None]:
        from psycopg import AsyncConnection

        key = int.from_bytes(
            blake2b(b"master-resume-turn:" + profile_id.bytes, digest_size=8).digest(),
            signed=True,
        )
        async with (
            await AsyncConnection.connect(self._conninfo, autocommit=True) as conn,
            conn.transaction(),
        ):
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
            yield
