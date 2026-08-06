from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import (
    AliasChoices,
    Field,
    HttpUrl,
    SecretStr,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


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
    CONSOLE = "console"
    JSON = "json"


class AppSettings(BaseSettings):
    name: str = "job-helper-agent"
    environment: Environment = Environment.LOCAL
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65_535)


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DB_",
        extra="ignore",
    )

    driver: str = "postgresql+asyncpg"

    host: str = "localhost"
    port: int = 5432

    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")
    database: str = "job_finder"

    echo: bool = False
    pool_pre_ping: bool = True
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def url(self) -> URL:
        return URL.create(
            drivername=self.driver,
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.database,
        )


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "openrouter"
    model: str = Field(
        default="google/gemini-2.5-flash",
        validation_alias=AliasChoices("LLM_MODEL", "PROCESSING_LLM"),
    )

    base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    api_key: SecretStr = Field(
        validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY"),
    )

    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)

    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )

    model_path: Path = Path(
        "~/projects/job-helper/.models/"
        "embeddinggemma-300M-Q8_0.gguf"
    )

    @property
    def resolved_model_path(self) -> Path:
        path = self.model_path.expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Embedding model not found: {path}"
            )

        return path


class AgentSettings(BaseSettings):
    supervisor_model: str = "local-supervisor"
    worker_model: str = "openrouter-worker"

    max_iterations: int = Field(default=10, ge=1)
    recursion_limit: int = Field(default=50, ge=1)

    enable_checker: bool = True
    parallel_workers: int = Field(default=4, ge=1)


class HhSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    user_agent: str = "job-helper/0.1"
    access_token: SecretStr | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0)


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
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    agents: AgentSettings = AgentSettings()
    hh: HhSettings = HhSettings()
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
