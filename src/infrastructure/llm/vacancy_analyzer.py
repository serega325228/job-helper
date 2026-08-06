import json

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage

from src.schemas.vacancy import (
    NormalizedVacancy,
    NormalizedVacancyBatch,
    RawVacancy,
)


class VacancyAnalyzer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(NormalizedVacancyBatch)

    async def normalize(
        self,
        raw_vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        payload = []
        for vacancy in raw_vacancies:
            item = vacancy.model_dump(mode="json", exclude={"fetched_at"})
            # The source payload carries reliable salary and location metadata,
            # while the description is already present in raw_text.
            item["raw_payload"].pop("description", None)
            item["raw_payload"].pop("branded_description", None)
            payload.append(item)
        messages = [
            SystemMessage(
                content=(
                    "Ты нормализуешь вакансии из разных источников. "
                    "Верни ровно одну запись для каждой входной записи и сохрани "
                    "source и external_id без изменений. Извлекай только явно "
                    "указанные факты, не заполняй отсутствующие данные догадками. "
                    "Hard-условия запиши в отдельные поля, смысловые требования, "
                    "навыки и обязанности — в soft_conditions."
                ),
            ),
            HumanMessage(
                content=(
                    "Вакансии:\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            ),
        ]

        result: NormalizedVacancyBatch = await self._model.ainvoke(messages)
        return result.items
