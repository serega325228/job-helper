from collections.abc import AsyncIterator, Iterator

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio.session import AsyncSession

from src.config.settings import EmbeddingSettings, Settings
from src.infrastructure.db.engine import Database
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.repositories.profile import ProfileRepository
from src.services.embedding import EmbeddingService
from src.services.profile import ProfileService


class ConfigProvider(Provider):
    @provide(scope=Scope.APP)
    def settings(self) -> Settings:
        return Settings()

class InfrastructureProvider(Provider):
    @provide(scope=Scope.APP)
    async def database(
        self,
        settings: Settings
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

    unit_of_work = provide(
        SqlAlchemyUnitOfWork,
        scope=Scope.REQUEST,
    )

class RepositoryProvider(Provider):
    profile_repository = provide(
        ProfileRepository,
        scope=Scope.REQUEST
    )

class ServiceProvider(Provider):
    profile_service = provide(
        ProfileService,
        scope=Scope.REQUEST,
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
