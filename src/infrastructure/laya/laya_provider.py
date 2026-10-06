import asyncio
from functools import cached_property

import laya


class LayaProvider:
    @cached_property
    def _agent(self):
        return laya.load("convaiinnovations/laya-multilingual")

    async def evaluate(
        self,
        state: dict,
        questions: dict,
    ) -> dict:
        return await asyncio.to_thread(
            self._agent.predict,
            state,
            questions,
            max_len=4096,
        )

    async def evaluate_batch(
        self,
        states: list[dict],
        questions: dict,
        batch_size: int = 32,
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._agent.predict_batch,
            states,
            questions,
            batch_size=batch_size,
            max_len=4096,
            sort_by_length=True,
        )
