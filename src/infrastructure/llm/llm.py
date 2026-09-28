import asyncio
import copy
import json
from structlog import get_logger
import os
import re
import tempfile
from pathlib import Path
from typing import Any, TypeVar, overload

import litellm
from openai import OpenAIError
from pydantic import BaseModel, SecretStr, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import LLMSettings, Settings
from src.exceptions.config import ConfigError, LLMError
from src.schemas.llm import (
    LOCAL_PROVIDERS,
    PROVIDERS,
    FeatureConfig,
    LLMConfig,
    LLMConfigRequest,
    LLMHealth,
    LLMTestRequest,
)
from src.services.sercurity import SecurityService

logger = get_logger()

_PROVIDER_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "azure_foundry": "azure_ai",
    "deepseek": "deepseek",
    "gemini": "gemini",
    "groq": "groq",
    "ollama": "ollama_chat",
    "openai_compatible": "openai",
    "openrouter": "openrouter",
}
_PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def get_model_name(config: LLMConfigRequest) -> str:
    prefix = _PROVIDER_PREFIXES[config.provider] + "/"
    model = config.model
    if config.provider == "ollama":
        model = model.removeprefix("ollama/")
    elif config.provider == "azure_foundry":
        model = model.removeprefix("azure/")
    return model if model.startswith(prefix) else prefix + model


def validate_llm_config(config: LLMConfigRequest) -> None:
    model = get_model_name(config)
    try:
        supported = litellm.get_supported_openai_params(
            model=model,
            custom_llm_provider=_PROVIDER_PREFIXES[config.provider],
        )
    except OpenAIError, ValueError, KeyError:
        supported = None
    if supported is not None:
        for field in ("temperature", "reasoning_effort"):
            if getattr(config, field) is not None and field not in supported:
                raise ConfigError("unsupported_parameter", field=field, status_code=422)
    info = (
        litellm.model_cost.get(model)
        or litellm.model_cost.get(model.removeprefix("openai/"))
        or {}
    )
    limit = info.get("max_output_tokens") or info.get("max_tokens")
    if isinstance(limit, int) and limit > 0 and config.max_tokens > limit:
        raise ConfigError(
            "token_budget_exceeds_model_limit", field="max_tokens", status_code=422
        )


class LLMConfigManager:
    def __init__(self, settings: Settings, security: SecurityService) -> None:
        self._settings = settings
        self._security = security
        self._path = settings.app.data_dir / "config.json"
        # ponytail: one process owns this file; multiple workers need a shared transactional store.
        self._lock = asyncio.Lock()
        try:
            self._stored = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._stored = {}
        except OSError, ValueError:
            raise ConfigError("config_file_unreadable", status_code=503) from None
        config, features = self._validate_stored(self._stored)
        self._settings.llm.config = config
        self._settings.features = features

    def _keys(self, stored: dict[str, Any]) -> dict[str, SecretStr]:
        encrypted = stored.get("api_keys", {})
        if not isinstance(encrypted, dict):
            raise ConfigError("invalid_key_store", status_code=503)
        keys: dict[str, SecretStr] = {}
        for provider, ciphertext in encrypted.items():
            name = "gemini" if provider == "google" else provider
            if name not in PROVIDERS or not isinstance(ciphertext, str):
                raise ConfigError("invalid_key_store", status_code=503)
            try:
                keys[name] = SecretStr(self._security.decrypt(ciphertext))
            except ValueError:
                raise ConfigError(
                    "api_key_decryption_failed", field=name, status_code=503
                ) from None
        return keys

    def _validate_stored(
        self, stored: dict[str, Any]
    ) -> tuple[LLMConfig | None, FeatureConfig | None]:
        if not isinstance(stored, dict):
            raise ConfigError("invalid_config_file", status_code=503)
        keys = self._keys(stored)
        try:
            raw_config = stored.get("llm")
            config = None
            if raw_config is not None:
                request = LLMConfigRequest.model_validate(raw_config)
                config = LLMConfig(
                    **request.model_dump(),
                    api_key=keys.get(request.provider),
                )
            raw_features = stored.get("features")
            features = (
                FeatureConfig.model_validate(raw_features)
                if raw_features is not None
                else None
            )
        except ValidationError:
            raise ConfigError("invalid_config_file", status_code=503) from None
        return config, features

    def _save(self, stored: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, filename = tempfile.mkstemp(
                dir=self._path.parent, prefix=".config."
            )
            temporary = Path(filename)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(stored, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        except OSError:
            raise ConfigError("config_save_failed", status_code=503) from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _commit(self, stored: dict[str, Any]) -> None:
        config, features = self._validate_stored(stored)
        self._save(stored)
        self._stored = stored
        self._settings.llm.config = config
        self._settings.features = features

    def get(self) -> LLMConfig | None:
        config = self._settings.llm.config
        return config.model_copy(deep=True) if config is not None else None

    def get_features(self) -> FeatureConfig | None:
        return self._settings.features

    def get_value(self, name: str, default: Any = None) -> Any:
        return copy.deepcopy(self._stored.get(name, default))

    def get_api_keys(self) -> dict[str, SecretStr]:
        return self._keys(self._stored)

    def resolve(self, request: LLMConfigRequest | LLMTestRequest) -> LLMConfig:
        override = request.api_key if isinstance(request, LLMTestRequest) else None
        key = (
            override
            if override is not None
            else self.get_api_keys().get(request.provider)
        )
        config = LLMConfig(**request.model_dump(exclude={"api_key"}), api_key=key)
        self.require_credentials(config)
        validate_llm_config(config)
        return config

    @staticmethod
    def require_credentials(config: LLMConfig) -> None:
        if config.provider not in LOCAL_PROVIDERS and (
            config.api_key is None or not config.api_key.get_secret_value().strip()
        ):
            raise ConfigError("api_key_required", field="api_key", status_code=422)

    async def update(self, request: LLMConfigRequest) -> LLMConfig:
        async with self._lock:
            config = self.resolve(request)
            self._commit(self._stored | {"llm": config.model_dump(mode="json")})
            return config

    async def update_features(self, features: FeatureConfig) -> FeatureConfig:
        async with self._lock:
            self._commit(self._stored | {"features": features.model_dump(mode="json")})
        return features

    async def update_values(self, values: dict[str, str]) -> None:
        if set(values) & {"llm", "features", "api_keys"}:
            raise ValueError("Use the typed configuration update methods")
        async with self._lock:
            self._commit(self._stored | values)

    async def update_api_keys(self, updates: dict[str, SecretStr | None]) -> None:
        async with self._lock:
            encrypted = dict(self._stored.get("api_keys", {}))
            if "google" in encrypted:
                encrypted.setdefault("gemini", encrypted.pop("google"))
            for provider, secret in updates.items():
                name = "gemini" if provider == "google" else provider
                if name not in PROVIDERS:
                    raise ConfigError(
                        "unsupported_provider", field="provider", status_code=422
                    )
                plaintext = (
                    secret.get_secret_value().strip() if secret is not None else ""
                )
                if plaintext:
                    encrypted[name] = self._security.encrypt(plaintext)
                else:
                    encrypted.pop(name, None)
            self._commit(self._stored | {"api_keys": encrypted})

    async def clear_api_keys(self) -> None:
        async with self._lock:
            self._commit(self._stored | {"api_keys": {}})


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class _HealthProbe(BaseModel):
    ok: bool


class LLMProvider:
    def __init__(self, settings: LLMSettings, config: LLMConfig | None) -> None:
        self._config = config.model_copy(deep=True) if config is not None else None
        self._timeout = settings.request_timeout_seconds
        self._max_retries = settings.max_retries

    @property
    def configured(self) -> bool:
        return self._config is not None and (
            self._config.provider in LOCAL_PROVIDERS
            or bool(
                self._config.api_key and self._config.api_key.get_secret_value().strip()
            )
        )

    @overload
    async def complete(
        self, prompt: str, system_prompt: str | None = None, *, schema: None = None
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
        return await self._complete(prompt, system_prompt, schema=schema)

    async def _complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        schema: type[ResponseModel] | None = None,
        token_limit: int | None = None,
    ) -> str | ResponseModel:
        config = self._config
        if not self.configured or config is None:
            raise ConfigError(field="llm")
        validate_llm_config(config)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        key = config.api_key.get_secret_value() if config.api_key else "not-required"
        request: dict[str, Any] = {
            "model": get_model_name(config),
            "custom_llm_provider": _PROVIDER_PREFIXES[config.provider],
            "messages": messages,
            "api_key": key,
            "api_base": str(config.api_base)
            if config.api_base
            else _PROVIDER_ENDPOINTS[config.provider],
            "max_tokens": min(config.max_tokens, token_limit)
            if token_limit
            else config.max_tokens,
            "timeout": self._timeout,
            "num_retries": 0,
            "max_retries": 0,
            "drop_params": False,
        }
        for field in ("temperature", "reasoning_effort", "api_version"):
            value = getattr(config, field)
            if value is not None:
                request[field] = value
        if schema is not None:
            request["response_format"] = schema
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (
                        litellm.Timeout,
                        litellm.RateLimitError,
                        litellm.APIConnectionError,
                        litellm.InternalServerError,
                    )
                ),
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential(multiplier=0.5, max=4),
                reraise=True,
            ):
                with attempt:
                    response = await litellm.acompletion(**request)
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise LLMError("output_truncated")
            content = self._extract_content(choice.message.content)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()
            if not content:
                raise LLMError("empty_output")
            return content if schema is None else schema.model_validate_json(content)
        except LLMError:
            raise
        except ValidationError:
            raise LLMError("invalid_structured_output") from None
        except OpenAIError as error:
            if isinstance(error, litellm.AuthenticationError):
                code = "authentication_failed"
            elif isinstance(error, litellm.Timeout):
                code = "timeout"
            elif isinstance(error, litellm.RateLimitError):
                code = "rate_limited"
            elif isinstance(error, litellm.UnsupportedParamsError):
                code = "unsupported_parameter"
            elif isinstance(error, litellm.BadRequestError):
                code = "invalid_provider_request"
            else:
                code = "provider_unavailable"
            logger.warning("LLM request failed: %s (%s)", code, type(error).__name__)
            raise LLMError(code) from None

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(getattr(part, "text", None), str):
                parts.append(part.text)
        return "\n".join(parts)

    async def check_llm_health(
        self, *, include_details: bool = False, test_prompt: str | None = None
    ) -> LLMHealth:
        config = self._config
        result = LLMHealth(
            healthy=False,
            provider=config.provider if config else None,
            model=config.model if config else None,
        )
        if not self.configured:
            result.error_code = "configuration_required"
            return result
        prompt = test_prompt or 'Return {"ok": true}.'
        if include_details:
            result.test_prompt = prompt
        try:
            response = await self._complete(
                prompt,
                "This is a connection test. Return JSON with ok=true.",
                schema=_HealthProbe,
                token_limit=256,
            )
            result.healthy = response.ok
            if not response.ok:
                result.error_code = "invalid_probe_response"
            if include_details:
                result.model_output = response.model_dump_json()
        except (LLMError, ConfigError) as error:
            result.error_code = error.code
        return result
