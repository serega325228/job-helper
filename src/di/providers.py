from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import AsyncGenerator

import httpx
from crawlee import ConcurrencySettings
from crawlee.configuration import Configuration
from crawlee.crawlers._playwright import PlaywrightCrawler
from crawlee.events import LocalEventManager
from crawlee.storage_clients._file_system import FileSystemStorageClient
from dishka import AsyncContainer, Provider, Scope, provide
from playwright.async_api import Playwright, async_playwright
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import (
    CrawleeSettings,
    ScrapingSettings,
    Settings,
    get_settings,
)
from src.exceptions.config import ConfigError
from src.infrastructure.db.engine import Database
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.embedding.llama_server import LlamaServerEmbedder
from src.infrastructure.laya.laya_provider import LayaProvider
from src.infrastructure.llm.llm import LLMConfigManager, LLMProvider
from src.infrastructure.llm.profile_analyzer import ProfileAnalyzer
from src.infrastructure.llm.vacancy_analyzer import VacancyAnalyzer
from src.infrastructure.pdf.pdf import PDFRender
from src.infrastructure.reranker.llama_server import LlamaServerReranker
from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
from src.ports.embedding import Embedder
from src.ports.reranker import Reranker
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.repositories.profile import ProfileRepository
from src.repositories.prompt import PromptRepository
from src.repositories.resume import ResumeRepository
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_match import VacancyMatchRepository
from src.schemas.llm import FeatureConfig
from src.services.cover_letter import CoverLetterService
from src.services.improver import ImproverService
from src.services.interview_prep import InterviewPrepService
from src.services.profile import ProfileService
from src.services.refiner import RefinerService
from src.services.scoring import ScoringService
from src.services.sercurity import SecurityService
from src.services.skill_canonicalization import SkillCanonicalizer
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService
from src.services.vacancy_preview import VacancyPreviewEvaluator
from src.services.vacancy_scraping import (
    CrawlerFactory,
    UnitOfWorkFactory,
    VacancyScrapingService,
    VacancyServiceFactory,
)


class ConfigProvider(Provider):
    @provide(scope=Scope.APP)
    def settings(self) -> Settings:
        return get_settings()


class InfrastructureProvider(Provider):
    @provide(scope=Scope.APP)
    async def database(
        self,
        settings: Settings,
    ) -> AsyncIterator[Database]:
        database = Database(settings.database)
        yield database
        await database.dispose()

    @provide(scope=Scope.REQUEST)
    async def session(
        self,
        database: Database,
    ) -> AsyncIterator[AsyncSession]:
        async with database.session() as session:
            yield session

    @provide(scope=Scope.APP)
    async def http_client(
        self,
        settings: Settings,
    ) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            timeout=settings.hh.request_timeout_seconds,
        ) as client:
            yield client

    @provide(scope=Scope.APP)
    def llm_config_manager(
        self, settings: Settings, security: SecurityService
    ) -> LLMConfigManager:
        return LLMConfigManager(settings, security)

    @provide(scope=Scope.REQUEST)
    def llm_provider(
        self, settings: Settings, config_manager: LLMConfigManager
    ) -> LLMProvider:
        return LLMProvider(settings.llm, config_manager.get())

    @provide(scope=Scope.REQUEST)
    def feature_config(self, config_manager: LLMConfigManager) -> FeatureConfig:
        features = config_manager.get_features()
        if features is None:
            raise ConfigError(field="features")
        return features

    @provide(scope=Scope.APP)
    def hh_client(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> HhApiClient:
        access_token = settings.hh.access_token
        return HhApiClient(
            http_client=http_client,
            user_agent=settings.hh.user_agent,
            access_token=(
                access_token.get_secret_value() if access_token is not None else None
            ),
        )

    unit_of_work = provide(
        SqlAlchemyUnitOfWork,
        scope=Scope.REQUEST,
    )

    @provide(scope=Scope.APP)
    def unit_of_work_factory(self, container: AsyncContainer) -> UnitOfWorkFactory:
        @asynccontextmanager
        async def open_unit_of_work() -> AsyncGenerator[SqlAlchemyUnitOfWork]:
            async with container() as scope:
                yield await scope.get(SqlAlchemyUnitOfWork)

        return open_unit_of_work

    hh_source = provide(HhVacancySource, scope=Scope.APP)
    linkedin_source = provide(LinkedInVacancySource, scope=Scope.APP)
    profile_analyzer = provide(ProfileAnalyzer, scope=Scope.REQUEST)
    vacancy_normalizer = provide(
        VacancyAnalyzer,
        scope=Scope.REQUEST,
        provides=VacancyNormalizer,
    )

    @provide(scope=Scope.APP)
    def skill_canonicalizer(self) -> SkillCanonicalizer:
        return SkillCanonicalizer()

    @provide(scope=Scope.APP)
    async def vacancy_reranker(self, settings: Settings) -> AsyncIterator[Reranker]:
        config = settings.reranker
        if config.backend == "local":
            from src.infrastructure.reranker.vacancy_reranker import VacancyReranker

            yield VacancyReranker(config.model_name, batch_size=config.batch_size)
        else:
            headers = (
                {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}
                if config.api_key
                else {}
            )
            async with httpx.AsyncClient(
                base_url=str(config.base_url),
                timeout=config.request_timeout_seconds,
                headers=headers,
            ) as client:
                yield LlamaServerReranker(
                    client, config.server_model_name, batch_size=config.batch_size
                )

    @provide(scope=Scope.APP)
    def embedding_service(self, settings: Settings) -> Iterator[Embedder]:
        config = settings.embedding
        if config.backend == "local":
            from src.infrastructure.embedding.embedding import EmbeddingService

            service = EmbeddingService(str(config.resolved_model_path))
            try:
                yield service
            finally:
                service.close()
        else:
            headers = (
                {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}
                if config.api_key
                else {}
            )
            with httpx.Client(
                base_url=str(config.base_url),
                timeout=config.request_timeout_seconds,
                headers=headers,
            ) as client:
                yield LlamaServerEmbedder(
                    client, config.server_model_name, batch_size=config.batch_size
                )

    @provide(scope=Scope.APP)
    async def playwright(self) -> AsyncIterator[Playwright]:
        pw = await async_playwright().start()
        yield pw
        await pw.stop()

    @provide(scope=Scope.APP)
    def pdf_render(self, playwright: Playwright, settings: Settings) -> PDFRender:
        return PDFRender(playwright, settings.pdf)

    @provide(scope=Scope.APP)
    def laya_provider(self) -> LayaProvider:
        return LayaProvider()

    @provide(scope=Scope.APP)
    def crawler_factory(
        self,
        settings: Settings,
    ) -> CrawlerFactory:
        crawlee_settings = settings.crawlee
        scraping_settings = settings.scraping

        @asynccontextmanager
        async def open_crawler() -> AsyncGenerator[PlaywrightCrawler]:
            with TemporaryDirectory(prefix="vacancy-crawl-") as storage_directory:
                configuration = Configuration(storage_dir=storage_directory)
                crawler = PlaywrightCrawler(
                    configuration=configuration,
                    storage_client=FileSystemStorageClient(),
                    event_manager=LocalEventManager.from_config(configuration),
                    configure_logging=False,
                    concurrency_settings=ConcurrencySettings(
                        min_concurrency=crawlee_settings.min_concurrency,
                        desired_concurrency=crawlee_settings.desired_concurrency,
                        max_concurrency=crawlee_settings.max_concurrency,
                        max_tasks_per_minute=crawlee_settings.max_requests_per_minute,
                    ),
                    max_requests_per_crawl=scraping_settings.max_detail_pages + 1,
                    max_request_retries=crawlee_settings.max_request_retries,
                    max_session_rotations=0,
                    retry_on_blocked=False,
                    ignore_http_error_status_codes=[404, 410],
                    navigation_timeout=timedelta(
                        seconds=crawlee_settings.navigation_timeout_seconds
                    ),
                    request_handler_timeout=timedelta(
                        seconds=crawlee_settings.request_timeout_seconds
                    ),
                    headless=crawlee_settings.headless,
                    fingerprint_generator=None,
                    use_incognito_pages=True,
                    goto_options={"wait_until": "domcontentloaded"},
                )
                try:
                    yield crawler
                finally:
                    request_queue = await crawler.get_request_manager()
                    await request_queue.drop()
                    key_value_store = await crawler.get_key_value_store()
                    await key_value_store.drop()

        return open_crawler


class RepositoryProvider(Provider):
    resume_repository = provide(ResumeRepository, scope=Scope.REQUEST)
    profile_repository = provide(ProfileRepository, scope=Scope.REQUEST)
    vacancy_repository = provide(VacancyRepository, scope=Scope.REQUEST)
    vacancy_match_repository = provide(VacancyMatchRepository, scope=Scope.REQUEST)
    prompt_repository = provide(PromptRepository, scope=Scope.APP)


class ServiceProvider(Provider):
    improver_service = provide(ImproverService, scope=Scope.REQUEST)
    interview_prep_service = provide(InterviewPrepService, scope=Scope.REQUEST)
    profile_service = provide(ProfileService, scope=Scope.REQUEST)
    vacancy_service = provide(VacancyService, scope=Scope.REQUEST)
    vacancy_preview_evaluator = provide(VacancyPreviewEvaluator, scope=Scope.APP)
    vacancy_match_service = provide(VacancyMatchService, scope=Scope.REQUEST)

    @provide(scope=Scope.APP)
    def vacancy_service_factory(
        self, container: AsyncContainer
    ) -> VacancyServiceFactory:
        @asynccontextmanager
        async def open_vacancy_service() -> AsyncGenerator[VacancyService]:
            async with container() as scope:
                yield await scope.get(VacancyService)

        return open_vacancy_service

    @provide(scope=Scope.APP)
    def vacancy_scraping_service(
        self,
        crawler_factory: CrawlerFactory,
        unit_of_work_factory: UnitOfWorkFactory,
        vacancy_service_factory: VacancyServiceFactory,
        evaluator: VacancyPreviewEvaluator,
        settings: Settings,
    ) -> VacancyScrapingService:
        return VacancyScrapingService(
            crawler_factory,
            unit_of_work_factory,
            vacancy_service_factory,
            evaluator,
            max_search_pages=settings.scraping.max_search_pages,
            max_previews=settings.scraping.max_previews,
            max_detail_pages=settings.scraping.max_detail_pages,
            laya_batch_size=settings.scraping.laya_batch_size,
        )

    @provide(scope=Scope.APP)
    def security_service(
        self,
        settings: Settings,
    ) -> SecurityService:
        return SecurityService(settings.app.data_dir)

    @provide(scope=Scope.REQUEST)
    def scoring_service(
        self,
        reranker: Reranker,
        embedding_service: Embedder,
        skill_canonicalizer: SkillCanonicalizer,
        laya: LayaProvider,
    ) -> ScoringService:
        return ScoringService(
            reranker,
            embedding_service,
            skill_canonicalizer,
            laya,
        )

    @provide(scope=Scope.REQUEST)
    def cover_letter(
        self, provider: LLMProvider, prompts: PromptRepository
    ) -> CoverLetterService:
        return CoverLetterService(
            provider,
            prompts,
        )

    @provide(scope=Scope.REQUEST)
    def refiner_service(
        self, provider: LLMProvider, settings: Settings, improver: ImproverService
    ) -> RefinerService:
        return RefinerService(
            provider,
            settings.refiner,
            improver,
        )
