from typing import Literal

from pydantic import BaseModel, HttpUrl, SecretStr

ReasoningEffortLiteral = Literal["minimal", "low", "medium", "high"]

class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: SecretStr
    api_base: HttpUrl | None = None
    reasoning_effort: ReasoningEffortLiteral | None = None
    api_version: str | None = None
