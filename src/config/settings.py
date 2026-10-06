from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    HttpUrl,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

from src.schemas.llm import FeatureConfig, LLMConfig


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
    reload: bool = False
    data_dir: Path = Path(__file__).parent.parent.parent / "data"


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
    database: str = "job_helper"

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


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )

    backend: Literal["local", "llama_server"] = "llama_server"
    base_url: HttpUrl = HttpUrl("http://localhost:8081")
    server_model_name: str = Field(default="Qwen3-Embedding-0.6B-Q8_0.gguf", min_length=1)
    api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    batch_size: int = Field(default=16, ge=1)
    model_path: Path = (
        Path(__file__).parent.parent.parent
        / ".models"
        #/ "embeddinggemma-300M-Q8_0.gguf"
        / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    )

    @property
    def resolved_model_path(self) -> Path:
        path = self.model_path.expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Embedding model not found: {path}")

        return path


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLM_",
        extra="ignore",
    )

    _config: LLMConfig | None = PrivateAttr(default=None)

    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @property
    def config(self) -> LLMConfig | None:
        return self._config

    @config.setter
    def config(self, value: LLMConfig | None) -> None:
        self._config = value

class TaskiqSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TASKIQ_",
        extra="ignore",
    )
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/"

    exchange_name: str = "job-helper"
    exchange_type: str = "direct"

    io_queue: str = "io"
    laya_queue: str = "laya"


class RefinerSettings(BaseSettings):
    """Configuration for refinement passes."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REFINER_",
        extra="ignore",
    )

    enable_keyword_injection: bool = Field(default=True)
    enable_ai_phrase_removal: bool = Field(default=True)
    enable_master_alignment_check: bool = Field(default=True)


class PDFSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PDF_",
        extra="ignore",
    )

    max_concurrency: int = Field(default=4, gt=0)


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=".env",
        extra="ignore",
    )

    backend: Literal["local", "llama_server"] = "llama_server"
    base_url: HttpUrl = HttpUrl("http://localhost:8082")
    server_model_name: str = Field(default="Qwen3-Reranker-0.6B-Q8_0.gguf", min_length=1)
    api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = Field(default=16, ge=1)
    candidate_limit: int = Field(default=40, ge=1)


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


class CrawleeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CRAWLEE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    min_concurrency: int = Field(default=1, ge=1)
    desired_concurrency: int = Field(default=3, ge=1)
    max_concurrency: int = Field(default=5, ge=1)
    max_requests_per_minute: int = Field(default=30, ge=1)
    max_request_retries: int = Field(default=2, ge=0)
    navigation_timeout_seconds: float = Field(default=30.0, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    headless: bool = True

    @model_validator(mode="after")
    def validate_concurrency(self) -> Self:
        if not self.min_concurrency <= self.desired_concurrency <= self.max_concurrency:
            raise ValueError(
                "Expected min_concurrency <= desired_concurrency <= max_concurrency"
            )
        return self


class ScrapingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCRAPING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_search_pages: int = Field(default=10, ge=1)
    max_previews: int = Field(default=250, ge=1)
    max_detail_pages: int = Field(default=50, ge=1)
    laya_batch_size: int = Field(default=32, ge=1)


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
    _features: FeatureConfig | None = PrivateAttr(default=None)
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    refiner: RefinerSettings = RefinerSettings()
    pdf: PDFSettings = PDFSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    agents: AgentSettings = AgentSettings()
    hh: HhSettings = HhSettings()
    taskiq: TaskiqSettings = TaskiqSettings()
    crawlee: CrawleeSettings = CrawleeSettings()
    scraping: ScrapingSettings = ScrapingSettings()
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
    def features(self) -> FeatureConfig | None:
        return self._features

    @features.setter
    def features(self, value: FeatureConfig | None) -> None:
        self._features = value

    @property
    def is_production(self) -> bool:
        return self.app.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.app.environment == Environment.TESTING


@lru_cache
def get_settings() -> Settings:
    return Settings()
