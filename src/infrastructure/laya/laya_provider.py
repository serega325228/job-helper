import laya


class LayaProvider:
    def __init__(self):
        self._agent = laya.load("convaiinnovations/laya-multilingual")

    def evaluate(
        self,
        state: dict,
        questions: dict,
    ) -> dict:
        return self._agent.predict(
            state,
            questions,
            max_len=4096, # hardcoded - bad
        )

    def evaluate_batch(
        self,
        states: list[dict],
        questions: dict,
        batch_size: int = 32,
    ) -> list[dict]:
        return self._agent.predict_batch(
            states,
            questions,
            batch_size=batch_size,
            max_len=4096,
            sort_by_length=True
        )
