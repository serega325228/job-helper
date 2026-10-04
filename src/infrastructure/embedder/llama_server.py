import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.ports.embedder import EMBEDDING_DIMENSIONS


class _EmbeddingResult(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    index: int = Field(ge=0)
    embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS
    )


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingResult]


class LlamaServerEmbedder:
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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset : offset + self._batch_size]
            response = await self._client.post(
                "v1/embeddings",
                json={
                    "model": self.model_name,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            results = sorted(
                _EmbeddingResponse.model_validate(response.json()).data,
                key=lambda result: result.index,
            )

            if [result.index for result in results] != list(range(len(batch))):
                raise ValueError(
                    "Embedding response must contain each input index once"
                )

            for result in results:
                if not any(result.embedding):
                    raise ValueError("Embedding response must contain nonzero vectors")
                vectors.append(result.embedding)
        return vectors
