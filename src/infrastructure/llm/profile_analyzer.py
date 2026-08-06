from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage

from src.schemas.profile import ProfileAnalysis


class ProfileAnalyzer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(ProfileAnalysis)

    async def analyze(self, raw_story: str) -> ProfileAnalysis:
        messages = [
            SystemMessage(
                content=(
                    "Ты анализируешь профессиональную историю кандидата. "
                    "Извлекай только факты, указанные пользователем. "
                    "Не придумывай отсутствующий опыт. Выдели подходящие "
                    "названия профессий и явно сформулированные предпочтения."
                ),
            ),
            HumanMessage(
                content=f"История кандидата:\n\n{raw_story}",
            ),
        ]

        return await self._model.ainvoke(messages)
