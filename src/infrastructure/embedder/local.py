from asyncio import to_thread
from pathlib import Path
from threading import Lock

from llama_cpp import (
    LLAMA_POOLING_TYPE_MEAN,
    Llama,
)


class EmbeddingService:
    def __init__(self, model_path: str) -> None:
        self.model_name = Path(model_path).stem
        self._model = Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_MEAN,
            n_ctx=2048,
            n_batch=2048,
            verbose=False,
        )
        # ponytail: one shared model serializes inference; use separate model workers to scale.
        self._lock = Lock()
        self._closed = False

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prompts = [f"task: search result | query: {text}" for text in texts]
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding service is closed")
            return to_thread(self._model.embed, prompts, normalize=True)

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
        prompts = [
            f"title: {title or 'none'} | text: {text}" for title, text in documents
        ]
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding service is closed")
            return to_thread(self._model.embed, prompts, normalize=True)

    def embed_vacancy(
        self,
        title: str,
        text: str,
    ) -> list[float]:
        return self.embed_document(text, title=title)

    def embed_vacancies(
        self,
        vacancies: list[tuple[str, str]],
    ) -> list[list[float]]:
        return self.embed_documents(vacancies)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._model.close()
                self._closed = True
