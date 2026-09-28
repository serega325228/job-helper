import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from dishka import Provider, Scope, make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError
from sqlalchemy.dialects import postgresql

from src.config.settings import AppSettings, LLMSettings, Settings
from src.exceptions.config import ConfigError, LLMError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.llm import LLMConfigManager, LLMProvider
from src.repositories.prompt import PromptRepository
from src.repositories.resume import ResumeRepository
from src.routers.config import (
    config_error_handler,
    config_validation_error_handler,
    llm_error_handler,
)
from src.routers.config import (
    router as config_router,
)
from src.routers.health import router as health_router
from src.schemas.llm import FeatureConfig, LLMConfigRequest
from src.services.sercurity import SecurityService


def settings_at(directory: str) -> Settings:
    return Settings(
        app=AppSettings(data_dir=Path(directory)),
        llm=LLMSettings(_env_file=None, max_retries=0),
        _env_file=None,
    )


def llm_request(**changes) -> LLMConfigRequest:
    return LLMConfigRequest.model_validate(
        {
            "provider": "openai_compatible",
            "model": "local-model",
            "api_base": "http://localhost:8001/v1",
            "max_tokens": 1000,
            "temperature": None,
        }
        | changes
    )


def feature_config() -> FeatureConfig:
    return FeatureConfig(
        enable_cover_letter=True,
        enable_outreach_message=False,
        enable_interview_prep=True,
        allow_unverified_skills=False,
    )


def application(settings, manager, unit_of_work):
    dependencies = Provider()
    dependencies.provide(lambda: settings, provides=Settings, scope=Scope.APP)
    dependencies.provide(lambda: manager, provides=LLMConfigManager, scope=Scope.APP)
    dependencies.provide(
        lambda: unit_of_work, provides=SqlAlchemyUnitOfWork, scope=Scope.REQUEST
    )

    def provider(config_manager: LLMConfigManager) -> LLMProvider:
        return LLMProvider(settings.llm, config_manager.get())

    dependencies.provide(provider, scope=Scope.REQUEST)
    container = make_async_container(dependencies)
    app = FastAPI()
    app.include_router(config_router)
    app.include_router(health_router)
    app.add_exception_handler(ConfigError, config_error_handler)
    app.add_exception_handler(LLMError, llm_error_handler)
    app.add_exception_handler(RequestValidationError, config_validation_error_handler)
    setup_dishka(container, app)
    return app, container


class RuntimeConfigTest(unittest.IsolatedAsyncioTestCase):
    async def test_restart_restores_settings_features_and_encrypted_keys(self):
        with TemporaryDirectory() as directory:
            settings = settings_at(directory)
            manager = LLMConfigManager(settings, SecurityService(Path(directory)))
            self.assertIsNone(settings.llm.config)
            await manager.update_api_keys({"google": SecretStr("provider-secret")})
            await manager.update(llm_request())
            await manager.update_features(feature_config())
            path = Path(directory) / "config.json"
            self.assertNotIn("provider-secret", path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            restored_settings = settings_at(directory)
            restored = LLMConfigManager(
                restored_settings, SecurityService(Path(directory))
            )
            self.assertEqual(restored_settings.llm.config, settings.llm.config)
            self.assertEqual(restored_settings.features, feature_config())
            self.assertEqual(
                restored.get_api_keys()["gemini"].get_secret_value(), "provider-secret"
            )

    async def test_failed_save_keeps_file_and_runtime_config(self):
        with TemporaryDirectory() as directory:
            settings = settings_at(directory)
            manager = LLMConfigManager(settings, SecurityService(Path(directory)))
            await manager.update(llm_request())
            path = Path(directory) / "config.json"
            before = path.read_bytes()
            original = manager.get()
            with (
                patch(
                    "src.infrastructure.llm.llm.os.replace",
                    side_effect=OSError("disk failure"),
                ),
                self.assertRaises(ConfigError),
            ):
                await manager.update(llm_request(model="other-model"))
            self.assertEqual(manager.get(), original)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob(".config.*")), [])

    async def test_concurrent_updates_preserve_other_sections_and_prompt_reads(self):
        with TemporaryDirectory() as directory:
            manager = LLMConfigManager(
                settings_at(directory), SecurityService(Path(directory))
            )
            prompts = PromptRepository(manager)
            self.assertEqual(
                prompts.get("cover_letter_prompt", "default"), ("default", False)
            )
            await asyncio.gather(
                manager.update_features(feature_config()),
                manager.update(llm_request()),
                manager.update_api_keys({"openai": SecretStr("test-key")}),
                manager.update_values(
                    {"cover_letter_prompt": "custom", "ui_language": "ru"}
                ),
            )
            self.assertEqual(
                prompts.get("cover_letter_prompt", "default"), ("custom", True)
            )
            restored = LLMConfigManager(
                settings_at(directory), SecurityService(Path(directory))
            )
            self.assertEqual(restored.get_value("ui_language"), "ru")
            self.assertEqual(restored.get_features(), feature_config())
            self.assertEqual(restored.get().model, "local-model")
            self.assertIn("openai", restored.get_api_keys())

    async def test_key_deletion_changes_new_requests_only(self):
        with TemporaryDirectory() as directory:
            settings = settings_at(directory)
            manager = LLMConfigManager(settings, SecurityService(Path(directory)))
            await manager.update_api_keys({"openai": SecretStr("test-key")})
            await manager.update(llm_request(provider="openai", api_base=None))
            first = LLMProvider(settings.llm, manager.get())
            await manager.update_api_keys({"openai": None})
            self.assertTrue(first.configured)
            self.assertFalse(LLMProvider(settings.llm, manager.get()).configured)

    def test_environment_cannot_populate_runtime_configuration(self):
        with TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "LLM_CONFIG": llm_request().model_dump_json(),
                    "LLM__CONFIG": llm_request().model_dump_json(),
                    "FEATURES": feature_config().model_dump_json(),
                },
            ):
                settings = settings_at(directory)
            self.assertIsNone(settings.llm.config)
            self.assertIsNone(settings.features)

    def test_invalid_configuration_and_unknown_fields(self):
        for change in [
            {"model": "  "},
            {"provider": "invalid"},
            {"max_tokens": True},
            {"max_tokens": "1000"},
            {"temperature": "0.2"},
            {"temperature": float("nan")},
            {"max_tokens": 0},
            {"unexpected": True},
            {"api_base": None},
            {"api_base": "http://user:secret@localhost/v1"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                llm_request(**change)
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(
                feature_config().model_dump() | {"enable_cover_letter": "false"}
            )
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate({"enable_cover_letter": False})

    def test_corrupt_file_and_unreadable_keys_are_not_overwritten(self):
        with TemporaryDirectory() as directory:
            security = SecurityService(Path(directory))
            path = Path(directory) / "config.json"
            for content in (
                "invalid json",
                json.dumps({"api_keys": {"openai": "bad-ciphertext"}}),
            ):
                path.write_text(content)
                with self.assertRaises(ConfigError):
                    LLMConfigManager(settings_at(directory), security)
                self.assertEqual(path.read_text(), content)

    async def test_config_and_health_routes(self):
        with TemporaryDirectory() as directory:
            settings = settings_at(directory)
            manager = LLMConfigManager(settings, SecurityService(Path(directory)))
            uow = SimpleNamespace(
                get_stats=AsyncMock(return_value={"has_master_resume": True})
            )
            app, container = application(settings, manager, uow)
            profile_id = uuid4()
            try:
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    self.assertIsNone((await client.get("/config/llm-api-key")).json())
                    self.assertIsNone((await client.get("/config/features")).json())
                    self.assertEqual(
                        (await client.get("/health")).json(), {"status": "healthy"}
                    )
                    with patch(
                        "src.infrastructure.llm.llm.litellm.acompletion",
                        new_callable=AsyncMock,
                    ) as complete:
                        status = (await client.get(f"/status/{profile_id}")).json()
                    complete.assert_not_called()
                    self.assertEqual(status["status"], "setup_required")
                    uow.get_stats.assert_awaited_with(profile_id)
                    bad = await client.put(
                        "/config/llm-api-key", json={"api_key": "must-stay-secret"}
                    )
                    self.assertEqual(bad.status_code, 422)
                    self.assertNotIn("must-stay-secret", bad.text)
                    saved = await client.put(
                        "/config/llm-api-key",
                        json=llm_request().model_dump(mode="json"),
                    )
                    self.assertEqual(saved.status_code, 200, saved.text)
                    self.assertEqual(saved.json()["api_key"], "")
                    flags = await client.put(
                        "/config/features", json=feature_config().model_dump()
                    )
                    self.assertEqual(flags.status_code, 200, flags.text)
                    language = await client.put(
                        "/config/language", json={"content_language": "ru"}
                    )
                    self.assertEqual(language.json()["content_language"], "ru")
                    template = "{job_description}\n{resume_data}\n{output_language}"
                    prompt = await client.put(
                        "/config/feature-prompts",
                        json={"cover_letter_prompt": template},
                    )
                    self.assertEqual(prompt.status_code, 200, prompt.text)
                    malformed = await client.put(
                        "/config/feature-prompts",
                        json={"cover_letter_prompt": "{resume_data.__class__}"},
                    )
                    self.assertEqual(malformed.status_code, 422)
                    before = (Path(directory) / "config.json").read_bytes()
                    model_response = SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content='{"ok":true}'),
                                finish_reason="stop",
                            )
                        ]
                    )
                    with patch(
                        "src.infrastructure.llm.llm.litellm.acompletion",
                        new_callable=AsyncMock,
                    ) as complete:
                        complete.return_value = model_response
                        test = await client.post(
                            "/config/llm-test",
                            json=llm_request(model="candidate").model_dump(mode="json"),
                        )
                        self.assertTrue(test.json()["healthy"], test.text)
                        self.assertEqual(
                            complete.await_args.kwargs["model"], "openai/candidate"
                        )
                        status = (await client.get(f"/status/{profile_id}")).json()
                        self.assertEqual(status["status"], "ready")
                    self.assertEqual(manager.get().model, "local-model")
                    self.assertEqual(
                        (Path(directory) / "config.json").read_bytes(), before
                    )
                    uow.get_stats.side_effect = RuntimeError("DB unavailable")
                    with patch(
                        "src.infrastructure.llm.llm.litellm.acompletion",
                        new_callable=AsyncMock,
                    ) as complete:
                        complete.return_value = model_response
                        status = (await client.get(f"/status/{profile_id}")).json()
                    self.assertEqual(status["status"], "degraded")
                    self.assertTrue(status["llm_healthy"])
                    self.assertFalse(status["database_healthy"])
            finally:
                await container.close()

    async def test_master_resume_query_contains_both_predicates(self):
        session = AsyncMock()
        session.scalar.return_value = True
        profile_id = uuid4()
        self.assertTrue(await ResumeRepository(session).has_master_resume(profile_id))
        statement = session.scalar.await_args.args[0].compile(
            dialect=postgresql.dialect()
        )
        self.assertIn("resumes.profile_id", str(statement))
        self.assertIn("resumes.resume_type", str(statement))
        self.assertIn(profile_id, statement.params.values())


if __name__ == "__main__":
    unittest.main()
