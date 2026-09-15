import logging
import re
from typing import Any, TypeVar, overload

import litellm
from litellm.router import Router
from litellm.types.router import RetryPolicy
from pydantic import BaseModel

from src.config.settings import LLMSettings

logger = logging.getLogger(__name__)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMProvider:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        litellm.drop_params = True
        litellm.modify_params = True
        self._model_name = self._get_model_name()
        self._router = self._build_router()
        self._router_config_key = self._config_fingerprint(settings)

    @overload
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: None = None,
    ) -> str: ...

    @overload
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: type[ResponseModel],
    ) -> ResponseModel: ...

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: type[ResponseModel] | None = None,
    ) -> str | ResponseModel:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": "primary",
            "messages": messages,
            "max_tokens": self._settings.max_tokens,
            "timeout": self._settings.request_timeout_seconds,
        }
        if self._supports_temperature():
            request["temperature"] = self._settings.temperature
        if self._settings.reasoning_effort is not None:
            request["reasoning_effort"] = self._settings.reasoning_effort
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

    def _build_router(self) -> Router:
        params: dict[str, Any] = {
            "model": self._model_name,
            "api_key": self._settings.api_key.get_secret_value(),
        }
        if self._settings.base_url is not None:
            params["api_base"] = self._normalize_api_base(
                str(self._settings.base_url),
            )
        if self._settings.api_version is not None:
            params["api_version"] = self._settings.api_version

        retries = self._settings.max_retries
        return Router(
            model_list=[
                {
                    "model_name": "primary",
                    "litellm_params": params,
                },
            ],
            num_retries=retries,
            retry_policy=RetryPolicy(
                AuthenticationErrorRetries=0,
                BadRequestErrorRetries=0,
                TimeoutErrorRetries=min(retries, 2),
                RateLimitErrorRetries=retries,
                ContentPolicyViolationErrorRetries=0,
                InternalServerErrorRetries=min(retries, 2),
            ),
            disable_cooldowns=True,
        )

    @staticmethod
    def _config_fingerprint(settings: LLMSettings) -> tuple[Any, ...]:
        return (
            settings.provider,
            settings.model,
            str(settings.base_url),
            settings.api_key.get_secret_value(),
            settings.api_version,
            settings.max_retries,
        )

    def _get_model_name(self) -> str:
        provider = self._settings.provider
        model = self._settings.model
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

    def _normalize_api_base(self, base_url: str) -> str:
        base_url = base_url.rstrip("/")
        if self._settings.provider in {"anthropic", "gemini", "openrouter"}:
            return base_url.removesuffix("/v1")
        if self._settings.provider == "ollama":
            for suffix in ("/api/generate", "/api/chat", "/api", "/v1"):
                if base_url.endswith(suffix):
                    return base_url.removesuffix(suffix)
        return base_url

    def _supports_temperature(self) -> bool:
        if self._model_name.startswith(("ollama/", "ollama_chat/")):
            return True
        try:
            model_info = litellm.get_model_info(model=self._model_name)
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
