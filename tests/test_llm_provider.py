import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
from pydantic import BaseModel, SecretStr

from src.config.settings import LLMSettings
from src.exceptions.config import ConfigError, LLMError
from src.infrastructure.llm.llm import LLMProvider, validate_llm_config
from src.schemas.llm import LLMConfig


class StructuredResponse(BaseModel):
    value: str


def config(model: str = "local-model", key: str = "first-key") -> LLMConfig:
    return LLMConfig(
        provider="openai_compatible",
        model=model,
        api_key=SecretStr(key),
        api_base="http://localhost:8001/v1",
        max_tokens=1000,
        temperature=None,
    )


def response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            ),
        ]
    )


class LLMProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_typed_output_uses_configured_parameters(self):
        provider = LLMProvider(LLMSettings(_env_file=None, max_retries=0), config())
        with patch(
            "src.infrastructure.llm.llm.litellm.acompletion", new_callable=AsyncMock
        ) as complete:
            complete.return_value = response('{"value":"ok"}')
            result = await provider.complete("prompt", schema=StructuredResponse)
        self.assertEqual(result, StructuredResponse(value="ok"))
        request = complete.await_args.kwargs
        self.assertEqual(request["api_key"], "first-key")
        self.assertEqual(request["api_base"], "http://localhost:8001/v1")
        self.assertEqual(request["max_tokens"], 1000)
        self.assertNotIn("temperature", request)
        self.assertFalse(request["drop_params"])

    async def test_requests_keep_independent_snapshots(self):
        settings = LLMSettings(_env_file=None, max_retries=0)
        settings.config = config()
        first = LLMProvider(settings, settings.config)
        started = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def complete(**request):
            requests.append(request)
            if request["model"] == "openai/local-model":
                started.set()
                await release.wait()
            return response(request["api_key"])

        with patch(
            "src.infrastructure.llm.llm.litellm.acompletion", side_effect=complete
        ):
            pending = asyncio.create_task(first.complete("first"))
            await started.wait()
            settings.config = config("second-model", "second-key")
            second = LLMProvider(settings, settings.config)
            self.assertEqual(await second.complete("second"), "second-key")
            release.set()
            self.assertEqual(await pending, "first-key")
        self.assertEqual(
            [item["model"] for item in requests],
            ["openai/local-model", "openai/second-model"],
        )

    async def test_missing_config_does_not_call_provider(self):
        provider = LLMProvider(LLMSettings(_env_file=None), None)
        with patch(
            "src.infrastructure.llm.llm.litellm.acompletion", new_callable=AsyncMock
        ) as complete:
            with self.assertRaises(ConfigError):
                await provider.complete("prompt")
            health = await provider.check_llm_health()
        complete.assert_not_called()
        self.assertFalse(health.healthy)
        self.assertEqual(health.error_code, "configuration_required")

    async def test_empty_invalid_and_truncated_responses(self):
        provider = LLMProvider(LLMSettings(_env_file=None, max_retries=0), config())
        for content, finish, code in [
            ("", "stop", "empty_output"),
            ("not json", "stop", "invalid_structured_output"),
            ('{"value":"ok"}', "length", "output_truncated"),
        ]:
            with self.subTest(code=code):
                with patch(
                    "src.infrastructure.llm.llm.litellm.acompletion",
                    new_callable=AsyncMock,
                ) as complete:
                    complete.return_value = response(content, finish)
                    with self.assertRaises(LLMError) as raised:
                        await provider.complete("prompt", schema=StructuredResponse)
                self.assertEqual(raised.exception.code, code)

    async def test_authentication_failure_is_not_retried_or_exposed(self):
        provider = LLMProvider(LLMSettings(_env_file=None, max_retries=2), config())
        error = litellm.AuthenticationError(
            "secret-key", llm_provider="openai", model="custom"
        )
        with patch(
            "src.infrastructure.llm.llm.litellm.acompletion", new_callable=AsyncMock
        ) as complete:
            complete.side_effect = error
            health = await provider.check_llm_health(include_details=True)
        self.assertEqual(complete.await_count, 1)
        self.assertEqual(health.error_code, "authentication_failed")
        self.assertNotIn("secret-key", health.model_dump_json())

    async def test_health_probe_checks_structured_output(self):
        provider = LLMProvider(LLMSettings(_env_file=None, max_retries=0), config())
        with patch(
            "src.infrastructure.llm.llm.litellm.acompletion", new_callable=AsyncMock
        ) as complete:
            complete.return_value = response('{"ok": true}')
            health = await provider.check_llm_health(include_details=True)
        self.assertTrue(health.healthy)
        self.assertEqual(complete.await_args.kwargs["max_tokens"], 256)

    def test_known_unsupported_parameters_are_rejected(self):
        candidate = config().model_copy(update={"temperature": 0.5})
        with (
            patch(
                "src.infrastructure.llm.llm.litellm.get_supported_openai_params",
                return_value=["max_tokens"],
            ),
            self.assertRaises(ConfigError) as raised,
        ):
            validate_llm_config(candidate)
        self.assertEqual(raised.exception.field, "temperature")


if __name__ == "__main__":
    unittest.main()
