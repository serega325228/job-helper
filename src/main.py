from contextlib import asynccontextmanager

from dishka.integrations.fastapi import setup_dishka as setup_fastapi_dishka
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from src.config.logging import configure_logging
from src.config.settings import get_settings
from src.di.container import create_container
from src.exceptions.config import ConfigError, LLMError
from src.infrastructure.llm.llm import LLMConfigManager
from src.infrastructure.taskiq.broker import io_broker, laya_broker
from src.routers.config import (
    config_error_handler,
    config_validation_error_handler,
    llm_error_handler,
)
from src.routers.config import router as config_router
from src.routers.health import router as health_router
from src.routers.vacancy import router as vacancy_router

settings = get_settings()

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.logging)
    await container.get(LLMConfigManager)
    await io_broker.startup()
    try:
        await laya_broker.startup()
        try:
            yield
        finally:
            await laya_broker.shutdown()
    finally:
        await io_broker.shutdown()
        await container.close()


app = FastAPI(
    title="Job Helper API",
    description="AI-powered job scraping, scoring and resume tailoring",
    lifespan=lifespan,
)

container = create_container()
app.add_exception_handler(ConfigError, config_error_handler)
app.add_exception_handler(LLMError, llm_error_handler)
app.add_exception_handler(RequestValidationError, config_validation_error_handler)
setup_fastapi_dishka(
    container=container,
    app=app,
)


# CORS middleware - origins configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(vacancy_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Job Helper API",
        "docs": "/docs",
    }


def main():
    """Entry point for the project.scripts console script."""
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
    )


if __name__ == "__main__":
    main()
