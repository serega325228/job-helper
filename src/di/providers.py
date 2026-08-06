from collections.abc import AsyncIterator, Iterator

import httpx
from dishka import Provider, Scope, provide
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import Settings
from src.infrastructure.db.engine import Database
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.profile_analyzer import ProfileAnalyzer
from src.infrastructure.llm.vacancy_analyzer import VacancyAnalyzer
from src.infrastructure.vacancy_sources.hh.client import HhApiClient
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.repositories.profile import ProfileRepository
from src.repositories.vacancy import VacancyRepository
from src.services.embedding import EmbeddingService
from src.services.profile import ProfileService
from src.services.vacancy import VacancyService


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
    def chat_model(self, settings: Settings) -> BaseChatModel:
        return ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=str(settings.llm.base_url),
            temperature=settings.llm.temperature,
            max_completion_tokens=settings.llm.max_tokens,
            timeout=settings.llm.request_timeout_seconds,
            max_retries=settings.llm.max_retries,
        )

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
                access_token.get_secret_value()
                if access_token is not None
                else None
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


class RepositoryProvider(Provider):
    profile_repository = provide(ProfileRepository, scope=Scope.REQUEST)
    vacancy_repository = provide(VacancyRepository, scope=Scope.REQUEST)


class ServiceProvider(Provider):
    profile_service = provide(ProfileService, scope=Scope.REQUEST)
    vacancy_service = provide(VacancyService, scope=Scope.REQUEST)

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
