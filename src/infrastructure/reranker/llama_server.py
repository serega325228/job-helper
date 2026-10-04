from collections.abc import Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field


class _RerankResult(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    index: int = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1)


class _RerankResponse(BaseModel):
    results: list[_RerankResult]


class LlamaServerReranker:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        model_name: str,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self._client = http_client
        self._batch_size = batch_size

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(documents), self._batch_size):
            batch = list(documents[offset : offset + self._batch_size])
            response = await self._client.post(
                "v1/rerank",
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": batch,
                    "top_n": len(batch),
                },
            )
            response.raise_for_status()
            results = sorted(
                _RerankResponse.model_validate(response.json()).results,
                key=lambda result: result.index,
            )
            if [result.index for result in results] != list(range(len(batch))):
                raise ValueError(
                    "Rerank response must contain each document index once"
                )
            scores.extend(result.relevance_score for result in results)
        return scores
