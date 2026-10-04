import math

import httpx
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DIMENSIONS = 1024

class _EmbeddingResult(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS)


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingResult]


class LlamaServerEmbedder:
    def __init__(
        self,
        http_client: httpx.Client,
        model_name: str,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self._client = http_client
        self._batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(
            [
                "Instruct: Given a job search query, retrieve relevant job vacancies"
                f"\nQuery: {text}"
                for text in texts
            ]
        )

    def embed_document(
        self,
        text: str,
        *,
        title: str | None = None,
    ) -> list[float]:
        return self.embed_documents([(title, text)])[0]

    def embed_documents(
        self,
        documents: list[tuple[str | None, str]],
    ) -> list[list[float]]:
        return self._embed(
            [f"{title}\n{text}" if title else text for title, text in documents]
        )

    def embed_vacancy(self, title: str, text: str) -> list[float]:
        return self.embed_document(text, title=title)

    def embed_vacancies(
        self,
        vacancies: list[tuple[str, str]],
    ) -> list[list[float]]:
        return self.embed_documents(vacancies)

    def _embed(self, prompts: list[str]) -> list[list[float]]:
        for offset in range(0, len(prompts), self._batch_size):
            batch = prompts[offset : offset + self._batch_size]
            response = self._client.post(
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

            #maybe delete this extra check
            if [result.index for result in results] != list(range(len(batch))):
                raise ValueError(
                    "Embedding response must contain each input index once"
                )

        return [result.embedding for result in results]
