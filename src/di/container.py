from dishka import AsyncContainer, make_async_container

from src.di.providers import (
    AgentProvider,
    ConfigProvider,
    InfrastructureProvider,
    RepositoryProvider,
    ServiceProvider,
)


def create_container() -> AsyncContainer:
    return make_async_container(
        ConfigProvider(),
        AgentProvider(),
        InfrastructureProvider(),
        RepositoryProvider(),
        ServiceProvider(),
    )
