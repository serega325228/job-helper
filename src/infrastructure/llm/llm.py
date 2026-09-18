import asyncio
import logging
import re
from typing import Any, Literal, TypeVar, overload

import litellm
from litellm.router import Router
from litellm.types.router import RetryPolicy
from pydantic import BaseModel, HttpUrl
from schemas.llm import LLMConfig

from src.config.settings import LLMSettings

logger = logging.getLogger(__name__)

class LLMConfigManager:
    def __init__(self, settings: LLMSettings, router: Router) -> None:
        self._settings = settings
        self._router = router
        self._lock = asyncio.Lock()

    async def update(self, config: LLMConfig):
        model_list = self._build_model_list(config)
        async with self._lock:
            self._router.set_model_list(model_list)
            self._settings.config = config

    @staticmethod
    def _build_model_list(config: LLMConfig) -> list:
        params = {
            "model_name": "primary",
            "litellm_params": {
                "model": config.model,
                "api_key": config.api_key,
            }
        }
        if config.api_base:
            params["litellm_params"]["api_base"] = config.api_base
        if config.reasoning_effort:
            params["litellm_params"]["reasoning_effort"] = config.reasoning_effort
        return [params]

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

class LLMProvider:
    def __init__(self, settings: LLMSettings, router: Router) -> None:
        self._settings = settings
        litellm.drop_params = True
        litellm.modify_params = True
        self._router = router

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
        if self._supports_temperature():
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
            #logger.exception("LLM completion failed for model %s", self._model_name)
            raise ValueError("LLM completion failed") from error

    """
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
    """

    def _supports_temperature(self) -> bool:
        if self._settings.config.model.startswith(("ollama/", "ollama_chat/")):
            return True
        try:
            model_info = litellm.get_model_info(model=self._settings.config.model)
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
