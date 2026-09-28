from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr

from src.schemas.llm import FeatureConfig, LLMConfigRequest

# Config Models


class LLMConfigResponse(LLMConfigRequest):
    """Response for LLM configuration."""

    api_key: str


FeatureConfigRequest = FeatureConfig
FeatureConfigResponse = FeatureConfig


class LanguageConfigRequest(BaseModel):
    """Request to update language settings."""

    ui_language: str | None = None  # en, es, zh, ja - for interface
    content_language: str | None = None  # en, es, zh, ja - for generated content


class LanguageConfigResponse(BaseModel):
    """Response for language settings."""

    ui_language: str = "en"  # Interface language
    content_language: str = "en"  # Generated content language
    supported_languages: list[str] = ["en", "es", "zh", "ja", "pt", "fr", "ko"]


class PromptOption(BaseModel):
    """Prompt option for resume tailoring."""

    id: str
    label: str
    description: str


class PromptConfigRequest(BaseModel):
    """Request to update prompt settings."""

    default_prompt_id: str | None = None


class PromptConfigResponse(BaseModel):
    """Response for prompt settings."""

    default_prompt_id: str
    prompt_options: list[PromptOption]


class FeaturePromptsRequest(BaseModel):
    """Request to update custom feature prompts.

    ``None`` means "don't change this field". An empty string clears the
    override — the server persists ``""`` so runtime resolution falls back
    to the built-in default without the key disappearing from config.json.
    """

    cover_letter_prompt: str | None = None
    outreach_message_prompt: str | None = None


class FeaturePromptsResponse(BaseModel):
    """Response for custom feature prompts.

    The ``*_default`` fields expose the built-in prompt strings so the UI
    can render them as placeholder text without duplicating the content
    across locales.
    """

    cover_letter_prompt: str
    outreach_message_prompt: str
    cover_letter_default: str
    outreach_message_default: str


# API Key Management Models
class ApiKeyProviderStatus(BaseModel):
    """Status of a single API key provider."""

    provider: str  # openai, anthropic, google, etc.
    configured: bool
    masked_key: str | None = None  # Shows last 4 chars if configured


class ApiKeyStatusResponse(BaseModel):
    """Response for API key status check."""

    providers: list[ApiKeyProviderStatus]


class ApiKeysUpdateRequest(BaseModel):
    """Request to update API keys."""

    model_config = ConfigDict(extra="forbid")

    openai: SecretStr | None = None
    azure_foundry: SecretStr | None = None
    anthropic: SecretStr | None = None
    gemini: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("gemini", "google")
    )
    openrouter: SecretStr | None = None
    deepseek: SecretStr | None = None
    groq: SecretStr | None = None
    openai_compatible: SecretStr | None = None
    ollama: SecretStr | None = None


class ApiKeysUpdateResponse(BaseModel):
    """Response after updating API keys."""

    message: str
    updated_providers: list[str]


# Update Cover Letter/Outreach Models
class UpdateCoverLetterRequest(BaseModel):
    """Request to update cover letter content."""

    content: str


class UpdateOutreachMessageRequest(BaseModel):
    """Request to update outreach message content."""

    content: str


class UpdateTitleRequest(BaseModel):
    """Request to update resume title."""

    title: str


class ResetDatabaseRequest(BaseModel):
    """Request to reset database with confirmation."""

    confirm: str | None = None


# health check
class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class StatusResponse(BaseModel):
    """Application status response."""

    status: str
    llm_configured: bool
    features_configured: bool
    llm_healthy: bool
    llm_error_code: str | None = None
    database_healthy: bool
    has_master_resume: bool
    database_stats: dict[str, Any]
