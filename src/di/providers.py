from collections.abc import AsyncIterator, Iterator

import httpx
from dishka import Provider, Scope, provide
from playwright.async_api import Playwright, async_playwright
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import Settings, get_settings
from src.exceptions.config import ConfigError
from src.infrastructure.db.engine import Database
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.laya.laya_provider import LayaProvider
from src.infrastructure.llm.llm import LLMConfigManager, LLMProvider
from src.infrastructure.llm.profile_analyzer import ProfileAnalyzer
from src.infrastructure.llm.vacancy_analyzer import VacancyAnalyzer
from src.infrastructure.pdf.pdf import PDFRender
from src.infrastructure.reranker.vacancy_reranker import VacancyReranker
from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.repositories.profile import ProfileRepository
from src.repositories.prompt import PromptRepository
from src.repositories.resume import ResumeRepository
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_match import VacancyMatchRepository
from src.schemas.llm import FeatureConfig
from src.services.cover_letter import CoverLetterService
from src.services.embedding import EmbeddingService
from src.services.improver import ImproverService
from src.services.interview_prep import InterviewPrepService
from src.services.profile import ProfileService
from src.services.refiner import RefinerService
from src.services.scoring import ScoringService
from src.services.sercurity import SecurityService
from src.services.skill_canonicalization import SkillCanonicalizer
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService


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
    hh_source = provide(HhVacancySource, scope=Scope.APP)
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
    def vacancy_reranker(self, settings: Settings) -> VacancyReranker:
        return VacancyReranker(
            settings.reranker.model_name,
            batch_size=settings.reranker.batch_size,
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
    vacancy_match_service = provide(VacancyMatchService, scope=Scope.REQUEST)

    @provide(scope=Scope.APP)
    def security_service(
        self,
        settings: Settings,
    ) -> SecurityService:
        return SecurityService(settings.app.data_dir)

    @provide(scope=Scope.REQUEST)
    def scoring_service(
        self,
        reranker: VacancyReranker,
        embedding_service: EmbeddingService,
        skill_canonicalizer: SkillCanonicalizer,
        laya: LayaProvider,
    ) -> ScoringService:
        return ScoringService(
            reranker,
            embedding_service,
            skill_canonicalizer,
            laya,
        )

    @provide(scope=Scope.APP)
    def embedding_service(
        self,
        settings: Settings,
    ) -> Iterator[EmbeddingService]:
        service = EmbeddingService(str(settings.embedding.resolved_model_path))
        try:
            yield service
        finally:
            service.close()

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
