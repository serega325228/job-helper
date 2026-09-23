from typing import Literal

from pydantic import BaseModel, HttpUrl, SecretStr


class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: SecretStr
    api_base: HttpUrl | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    api_version: str | None = None
