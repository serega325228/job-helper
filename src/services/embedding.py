from llama_cpp import (
    Llama,
    LLAMA_POOLING_TYPE_MEAN,
)


class EmbeddingService:
    def __init__(self, model_path: str) -> None:
        self._model = Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_MEAN,
            n_ctx=2048,
            n_batch=2048,
            verbose=False,
        )

    def embed_query(self, text: str) -> list[float]:
        prompt = f"task: search result | query: {text}"

        vectors = self._model.embed(
            [prompt],
            normalize=True,
        )
        return vectors[0]

    def embed_vacancy(
        self,
        title: str,
        text: str,
    ) -> list[float]:
        prompt = f"title: {title} | text: {text}"

        vectors = self._model.embed(
            [prompt],
            normalize=True,
        )
        return vectors[0]

    def embed_vacancies(
        self,
        vacancies: list[tuple[str, str]],
    ) -> list[list[float]]:
        prompts = [
            f"title: {title} | text: {text}"
            for title, text in vacancies
        ]

        return self._model.embed(
            prompts,
            normalize=True,
        )

    def close(self) -> None:
        self._model.close()
