from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.infrastructure.vacancy_sources.hh.schemas import NormalizedVacancy, NormalizedVacancyBatch, RawVacancy

class VacancyAnalyzer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(NormalizedVacancyBatch)

    async def normalize(self, raw_vacancies: list[RawVacancy]) -> list[NormalizedVacancy]:
        messages = [
            SystemMessage(
                content=(
                    "Ты анализируешь вакансию. "
                    "Извлекай только факты, указанные в вакансии. "
                    "Не придумывай отсутствующие данные."
                ),
            ),
            HumanMessage(
                content=f"Вакансии:\n\n{raw_vacancies}",
            ),
        ]

        return await self._model.ainvoke(messages).vacancies
