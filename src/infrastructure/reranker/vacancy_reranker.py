import asyncio
from collections.abc import Sequence

import torch
from sentence_transformers import CrossEncoder


class VacancyReranker:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
    ) -> None:
        self._model = CrossEncoder(
            model_name,
            activation_fn=torch.nn.Sigmoid(),
        )
        self._batch_size = batch_size

    async def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        if not documents:
            return []

        pairs = [
            (query, document)
            for document in documents
        ]

        scores = await asyncio.to_thread(
            self._model.predict,
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        return [
            float(score)
            for score in scores
        ]
