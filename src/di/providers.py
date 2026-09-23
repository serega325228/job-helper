from collections.abc import AsyncIterator, Iterator

import httpx
from dishka import Provider, Scope, provide
from infrastructure.pdf.pdf import PDFRender
from litellm import RetryPolicy
from litellm.router import Router
from playwright.async_api import Browser, Playwright, async_playwright
from services.refiner import RefinerService
from sqlalchemy.ext.asyncio import AsyncSession

from repositories.prompt import PromptRepository
from services.cover_letter import CoverLetterService
from services.sercurity import SecurityService
from src.config.settings import Settings
from src.infrastructure.db.engine import Database
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.llm import LLMConfigManager, LLMProvider, build_model_list
from src.infrastructure.llm.profile_analyzer import ProfileAnalyzer
from src.infrastructure.llm.vacancy_analyzer import VacancyAnalyzer
from src.infrastructure.reranker.vacancy_reranker import VacancyReranker
from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.repositories.profile import ProfileRepository
from src.repositories.vacancy import VacancyRepository
from src.repositories.vacancy_match import VacancyMatchRepository
from src.services.embedding import EmbeddingService
from src.services.profile import ProfileService
from src.services.scoring import ScoringService
from src.services.skill_canonicalization import SkillCanonicalizer
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService


class ConfigProvider(Provider):
    @provide(scope=Scope.APP)
    def settings(self) -> Settings:
        return Settings()


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
    async def llm_config_manager(self, settings: Settings, router: Router) -> LLMConfigManager:
        return LLMConfigManager(
            settings.llm,
            router,
        )

    @provide(scope=Scope.APP)
    async def llm_router(self, settings: Settings) -> Router:
        config = settings.llm.config
        model_list = []
        if config is not None:
            model_list = build_model_list(config)

        retries = settings.llm.max_retries
        return Router(
            model_list=model_list,
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

    @provide(scope=Scope.REQUEST)
    def llm_provider(self, settings: Settings, router: Router) -> LLMProvider:
        return LLMProvider(settings.llm, router)

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
    profile_analyzer = provide(ProfileAnalyzer, scope=Scope.APP)
    vacancy_normalizer = provide(
        VacancyAnalyzer,
        scope=Scope.APP,
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


class RepositoryProvider(Provider):
    profile_repository = provide(ProfileRepository, scope=Scope.REQUEST)
    vacancy_repository = provide(VacancyRepository, scope=Scope.REQUEST)
    vacancy_match_repository = provide(VacancyMatchRepository, scope=Scope.REQUEST)
    prompt_repository = provide(PromptRepository, scope=Scope.APP)


class ServiceProvider(Provider):
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
    ) -> ScoringService:
        return ScoringService(
            reranker,
            embedding_service,
            skill_canonicalizer,
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
        self, provider: LLMProvider, settings: Settings
    ) -> RefinerService:
        return RefinerService(
            provider,
            settings.refiner
        )
