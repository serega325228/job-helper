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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await to_thread(self._embed, texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding service is closed")
            return self._model.embed(texts, normalize=True)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._model.close()
                self._closed = True
