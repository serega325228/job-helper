import asyncio
import re
from typing import Any, Literal, TypeVar, overload

import litellm
from litellm.router import Router
from litellm.types.router import RetryPolicy
from pydantic import BaseModel, HttpUrl
from schemas.llm import LLMConfig
from structlog import get_logger

from src.config.settings import LLMSettings

logger = get_logger()

class ConfigError(Exception):
    """Custom excpetion for missing/empty config"""
    pass

def get_model_name(config: LLMConfig) -> str:
    provider = config.provider
    model = config.model
    prefixes = {
        "anthropic": "anthropic/",
        "azure_foundry": "azure_ai/",
        "deepseek": "deepseek/",
        "gemini": "gemini/",
        "groq": "groq/",
        "ollama": "ollama_chat/",
        "openai_compatible": "openai/",
        "openrouter": "openrouter/",
    }
    prefix = prefixes.get(provider, "")
    aliases = {
        "azure_foundry": ("azure/",),
        "ollama": ("ollama/",),
    }
    if not prefix or model.startswith((prefix, *aliases.get(provider, ()))):
        return model
    return f"{prefix}{model}"

def build_model_list(config: LLMConfig) -> list:
    params = {
        "model": get_model_name(config),
        "api_key": config.api_key,
    }

    if config.api_base:
        params["api_base"] = config.api_base
    if config.reasoning_effort:
        params["reasoning_effort"] = config.reasoning_effort
    if config.api_version:
        params["api_version"] = config.api_version

    model_list = {
        "model_name": "primary",
        "litellm_params": params
    }

    return [model_list]

class LLMConfigManager:
    def __init__(self, settings: LLMSettings, router: Router) -> None:
        self._settings = settings
        self._router = router
        self._lock = asyncio.Lock()

    async def update(self, config: LLMConfig):
        model_list = build_model_list(config)
        async with self._lock:
            self._router.set_model_list(model_list)
            self._settings.config = config

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

class LLMProvider:
    def __init__(self, settings: LLMSettings, router: Router) -> None:
        self._settings = settings
        litellm.drop_params = True
        litellm.modify_params = True
        self._router = router
        self._model_name = get_model_name(settings.config) if settings.config else ""

    @overload
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str: ...

    @overload
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: type[ResponseModel],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ResponseModel: ...

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: type[ResponseModel] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str | ResponseModel:
        if self._settings.config is None:
            raise ConfigError()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request = {
            "model": "primary",
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": self._settings.request_timeout_seconds,
        }
        if self._supports_temperature(self._model_name):
            request["temperature"] = temperature
        if self._settings.config.reasoning_effort:
            request["reasoning_effort"] = self._settings.config.reasoning_effort
        if schema is not None:
            request["response_format"] = schema

        try:
            response = await self._router.acompletion(**request)
            content = self._extract_content(response.choices[0].message.content)
            content = self._strip_thinking_tags(content)
            if not content:
                raise ValueError("LLM returned an empty response")

            if schema is None:
                return content
            return schema.model_validate_json(content)
        except Exception as error:
            logger.exception("LLM completion failed for model %s", self._model_name)
            raise ValueError("LLM completion failed") from error

    # @staticmethod
    # def _normalize_api_base(config: LLMConfig) -> str:
    #     base_url = config.api_base.rstrip("/")
    #     if config.provider in {"anthropic", "gemini", "openrouter"}:
    #         return base_url.removesuffix("/v1")
    #     if config.provider == "ollama":
    #         for suffix in ("/api/generate", "/api/chat", "/api", "/v1"):
    #             if base_url.endswith(suffix):
    #                 return base_url.removesuffix(suffix)
    #     return base_url

    @staticmethod
    def _supports_temperature(model_name: str) -> bool:
        if model_name.startswith(("ollama/", "ollama_chat/")):
            return True
        try:
            model_info = litellm.get_model_info(model=model_name)
        except Exception:
            return False
        supported_params = model_info.get("supported_openai_params", [])
        return "temperature" in supported_params

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(getattr(part, "text", None), str):
                parts.append(part.text)
        return "\n".join(parts)

    @staticmethod
    def _strip_thinking_tags(content: str) -> str:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()

    def get_safe_max_tokens(
        self,
        requested: int | None = None,
    ) -> int:
        """Return a token count safe for the given model, clamped to its output limit.

        Queries LiteLLM's model registry for ``max_output_tokens`` and returns
        ``min(requested, model_limit)`` so callers never send a value that exceeds
        what the backend actually supports.

        If the model is not in the registry (e.g. custom Ollama models), it falls
        back to a conservative limit. The verified OpenCode Zen HY3 route is an
        exception because JSON extraction disables reasoning for that model and
        needs the full structured-output budget.

        Args:
            model_name: LiteLLM-formatted model name (from get_model_name).
            requested: Desired token budget; defaults to DEFAULT_JSON_MAX_TOKENS.
            config: Optional provider configuration for scoped compatibility rules.

        Returns:
            Safe token count, clamped correctly and always >= 1.
        """
        config = self.get_config()
        if config is None:
            raise ConfigError()

        if not requested:
            return self._settings.max_tokens

        safe_requested = max(1, requested)

        model_name = get_model_name(config)

        try:
            info = litellm.get_model_info(model=model_name)
            model_limit = info.get("max_output_tokens") or info.get("max_tokens")
            if model_limit and isinstance(model_limit, int) and model_limit > 0:
                safe = min(safe_requested, model_limit)
                if safe < safe_requested:
                    logger.debug(
                        "max_tokens clamped %d → %d for model %s (model limit)",
                        safe_requested,
                        safe,
                        model_name,
                    )
                return safe
        except Exception:
            pass  # Model not in registry, drop down to fallback logic

        safe = min(safe_requested, self._settings.max_tokens)
        logger.debug(
            "Model %s not in LiteLLM registry, using fallback max_tokens %d",
            model_name,
            safe,
        )
        return safe

    async def check_llm_health(
        self,
        *,
        include_details: bool = False,
        test_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Check if the LLM provider is accessible and working."""
        # if config is None:
        #     config = get_llm_config()

        # Check if API key is configured. Ollama and openai_compatible local
        # servers often run without auth, so a blank key is acceptable for those
        # providers — a sentinel is passed downstream (see _effective_api_key)
        # to satisfy the OpenAI client's non-empty-string validation.
        config = self.get_config()
        if config is None:
            return {
                "healthy": False,
                "error_code": "config_missing",
            }
        if config.provider not in ("ollama", "openai_compatible") and not config.api_key:
            return {
                "healthy": False,
                "provider": config.provider,
                "model": config.model,
                "error_code": "api_key_missing",
            }

        prompt = test_prompt or "Hi"

        try:
            # Make a minimal test call with timeout
            # Pass API key directly to avoid race conditions with global os.environ
            content = await self.complete(
                prompt,
                max_tokens=64,
            )
            if not content:
                # LLM-003: Empty response (even after reasoning_content / thinking
                # fallbacks in _extract_choice_text) marks health as unhealthy.
                logger.warning(
                    "LLM health check returned empty content",
                    extra={"provider": config.provider, "model": config.model},
                )
                result: dict[str, Any] = {
                    "healthy": False,
                    "provider": config.provider,
                    "model": config.model,
                    "error_code": "empty_content",
                    "message": "LLM returned empty response",
                }
                if include_details:
                    result["test_prompt"] = prompt
                    result["model_output"] = None
                return result

            result = {
                "healthy": True,
                "provider": config.provider,
                "model": config.model,
            }
            if include_details:
                result["test_prompt"] = prompt
                result["model_output"] = content
                # Surface reasoning/thinking text separately ONLY when the model
                # also returned distinct primary content. If message.content was
                # empty, _extract_choice_text already folded the reasoning text
                # into `content` above — surfacing it here too would duplicate
                # identical text in "Model output" and "Model thinking".
                # msg = response.choices[0].message
                # primary_content = _join_text_parts(
                #     _extract_text_parts(_safe_get(msg, "content"))
                # )
                # reasoning_text = None
                # if primary_content:
                #     reasoning_text = (
                #         _join_text_parts(_extract_text_parts(_safe_get(msg, "reasoning_content")))
                #         or _join_text_parts(_extract_text_parts(_safe_get(msg, "thinking")))
                #     )
                # result["reasoning_content"] = (
                #     _to_code_block(reasoning_text) if reasoning_text else None
                # )
            return result
        except Exception as e:
            # Log full exception details server-side, but do not expose them to clients
            logger.exception(
                "LLM health check failed",
                extra={"provider": config.provider, "model": config.model},
            )

            # Provide a minimal, actionable client-facing hint without leaking secrets.
            error_code = "health_check_failed"
            message = str(e)
            if "404" in message and "/v1/v1/" in message:
                error_code = "duplicate_v1_path"
            elif "404" in message:
                error_code = "not_found_404"
            elif "<!doctype html" in message.lower() or "<html" in message.lower():
                error_code = "html_response"
            result = {
                "healthy": False,
                "provider": config.provider,
                "model": config.model,
                "error_code": error_code,
            }
            if include_details:
                result["test_prompt"] = prompt
                result["model_output"] = None
                # Scrub api-key-like tokens before surfacing the upstream error
                # text so the Settings UI can't be used to read back even a
                # partially-masked copy of the configured key.
                #result["error_detail"] = _to_code_block(_scrub_secrets(message))
            return result

    def get_config(self) -> LLMConfig | None:
        return self._settings.config
