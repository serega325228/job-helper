from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    StrictBool,
    field_validator,
    model_validator,
)

ProviderName = Literal[
    "openai",
    "azure_foundry",
    "anthropic",
    "gemini",
    "openrouter",
    "deepseek",
    "groq",
    "openai_compatible",
    "ollama",
]
PROVIDERS = (
    "openai",
    "azure_foundry",
    "anthropic",
    "gemini",
    "openrouter",
    "deepseek",
    "groq",
    "openai_compatible",
    "ollama",
)
LOCAL_PROVIDERS = {"ollama", "openai_compatible"}
ReasoningEffortLiteral = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "default"
]


class LLMConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName
    model: str = Field(min_length=1, strict=True)
    max_tokens: int = Field(gt=0, strict=True)
    temperature: float | None = Field(ge=0, le=2, strict=True, allow_inf_nan=False)
    api_base: HttpUrl | None = None
    reasoning_effort: ReasoningEffortLiteral | None = None
    api_version: str | None = Field(default=None, min_length=1, strict=True)

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: object) -> object:
        return "gemini" if value == "google" else value

    @field_validator("model", "api_version", mode="before")
    @classmethod
    def strip_identifier(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        if (
            self.provider in LOCAL_PROVIDERS | {"azure_foundry"}
            and self.api_base is None
        ):
            raise ValueError("api_base is required for this provider")
        if self.api_base and (self.api_base.username or self.api_base.password):
            raise ValueError("api_base must not contain credentials")
        return self


class LLMConfig(LLMConfigRequest):
    api_key: SecretStr | None = Field(default=None, exclude=True)


class LLMTestRequest(LLMConfigRequest):
    api_key: SecretStr | None = None


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enable_cover_letter: StrictBool
    enable_outreach_message: StrictBool
    enable_interview_prep: StrictBool
    allow_unverified_skills: StrictBool


class LLMHealth(BaseModel):
    healthy: bool
    provider: str | None = None
    model: str | None = None
    error_code: str | None = None
    test_prompt: str | None = None
    model_output: str | None = None
