from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class ProfileAnalysis(BaseModel):
    summary: str
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

class ProfileAnalyzer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(ProfileAnalysis)

    async def analyze(self, raw_story: str) -> ProfileAnalysis:
        messages = [
            SystemMessage(
                content=(
                    "Ты анализируешь профессиональную историю кандидата. "
                    "Извлекай только факты, указанные пользователем. "
                    "Не придумывай отсутствующий опыт."
                ),
            ),
            HumanMessage(
                content=f"История кандидата:\n\n{raw_story}",
            ),
        ]

        return await self._model.ainvoke(messages)
