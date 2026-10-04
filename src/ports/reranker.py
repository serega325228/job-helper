from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    model_name: str

    async def score(self, query: str, documents: Sequence[str]) -> list[float]: ...
