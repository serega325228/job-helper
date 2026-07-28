from dishka import AsyncContainer, make_async_container

from app.di.providers import (
    ApplicationProvider,
    ConfigProvider,
    InfrastructureProvider,
)


def create_container() -> AsyncContainer:
    return make_async_container(
        ConfigProvider(),
        InfrastructureProvider(),
        ApplicationProvider(),
    )
