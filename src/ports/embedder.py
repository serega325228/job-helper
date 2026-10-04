from typing import Protocol

EMBEDDING_DIMENSIONS = 1024


class Embedder(Protocol):
    model_name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
