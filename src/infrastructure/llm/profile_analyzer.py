from src.infrastructure.llm.llm import LLMProvider
from src.schemas.profile import ProfileAnalysis


class ProfileAnalyzer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def analyze(self, raw_story: str) -> ProfileAnalysis:
        system_prompt = (
            "Ты анализируешь профессиональную историю кандидата. "
            "Извлекай только факты, указанные пользователем. "
            "Не придумывай отсутствующий опыт. Отделяй подтверждённые "
            "навыки и опыт кандидата от его поисковых preference_intents. "
            "Создавай intent только для явно названного направления поиска."
        )
        return await self._llm.complete(
            f"История кандидата:\n\n{raw_story}",
            system_prompt,
            schema=ProfileAnalysis,
        )
