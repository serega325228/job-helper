from typing import Protocol


class Embedder(Protocol):
    model_name: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(
        self,
        documents: list[tuple[str | None, str]],
    ) -> list[list[float]]: ...
