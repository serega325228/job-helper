from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import (
    Field,
    HttpUrl,
    SecretStr,
    computed_field,
)

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(StrEnum):
    CONSOLE = "CONSOLE"
    JSON = "JSON"


class AppSettings(BaseSettings):
    name: str = "job-helper-agent"
    environment: Environment = Environment.LOCAL
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65_535)


class DatabaseSettings(BaseSettings):
    path: Path = Path("data/app.db")
    echo: bool = False

    @property
    def url(self) -> str:
        absolute_path = self.path.resolve()
        return f"sqlite+aiosqlite:///{absolute_path.as_posix()}"


class LLMSettings(BaseSettings):
    provider: str = "openrouter"
    model: str = "google/gemini-2.5-flash"

    base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    api_key: SecretStr

    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)

    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class AgentSettings(BaseSettings):
    supervisor_model: str = "local-supervisor"
    worker_model: str = "openrouter-worker"

    max_iterations: int = Field(default=10, ge=1)
    recursion_limit: int = Field(default=50, ge=1)

    enable_checker: bool = True
    parallel_workers: int = Field(default=4, ge=1)


class LoggingSettings(BaseSettings):
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.CONSOLE

    log_to_file: bool = True
    directory: Path = Path("logs")
    filename: str = "app.log"

    max_bytes: int = Field(
        default=1024 * 1024 * 10,
        ge=1,
    )
    backup_count: int = Field(default=5, ge=0)

class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    database: DatabaseSettings
    llm: LLMSettings
    agents: AgentSettings = AgentSettings()
    logging: LoggingSettings = LoggingSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    @property
    def is_production(self) -> bool:
        return self.app.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.app.environment == Environment.TESTING


@lru_cache
def get_settings() -> Settings:
    return Settings()
