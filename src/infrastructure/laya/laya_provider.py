import asyncio
from functools import cached_property

import laya


class LayaProvider:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @cached_property
    def _agent(self):
        return laya.load("convaiinnovations/laya-multilingual")

    async def _run(self, predict):
        async with self._lock:
            inference = asyncio.create_task(asyncio.to_thread(predict))
            try:
                return await asyncio.shield(inference)
            except asyncio.CancelledError:
                await inference
                raise

    async def load(self) -> None:
        await self._run(lambda: self._agent)

    async def evaluate(self, state: dict, questions: dict) -> dict:
        return await self._run(
            lambda: self._agent.predict(state, questions, max_len=4096)
        )

    async def evaluate_batch(
        self, states: list[dict], questions: dict, batch_size: int = 32
    ) -> list[dict]:
        return await self._run(
            lambda: self._agent.predict_batch(
                states,
                questions,
                batch_size=batch_size,
                max_len=4096,
                sort_by_length=True,
            )
        )
