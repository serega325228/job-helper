from contextlib import asynccontextmanager

import structlog
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph.state import CompiledStateGraph
from pydantic.dataclasses import dataclass
from src.graphs.supervisor.tools import SupervisorToolHandlers

from config.logging import configure_logging
from di.container import create_container
from src.agents.supervisor.tools import create_supervisor_tools
from src.config.settings import get_settings

settings = get_settings()

configure_logging(settings.logging)

logger = structlog.get_logger().bind(component=__name__)

repositories = create_repositories(settings)
services = create_services(repositories, settings)
models = create_model_registry(settings)

profile_graph = create_profile_graph(models.worker, services)
search_graph = create_search_graph(services)
resume_graph = create_resume_graph(services)
matching_graph = create_matching_graph(services)
preparation_graph = create_preparation_graph(services)

handlers = SupervisorToolHandlers(
    profile_graph=profile_graph,
    search_graph=search_graph,
    resume_graph=resume_graph,
    matching_graph=matching_graph,
    preparation_graph=preparation_graph,
)

tools = create_supervisor_tools(handlers)
supervisor = create_supervisor_agent(models.supervisor, tools)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Migrate DB here
    yield
    # Shutdown - wrap each cleanup in try-except to ensure all resources are released
    try:
        await close_pdf_renderer()
    except Exception as e:
        logger.error(f"Error closing PDF renderer: {e}")

    try:
        await db.close()
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    try:
        await container.close()
    except Exception as e:
        logger.error(f"Error closing DI container: {e}")


app = FastAPI(
    title="Job Helper API",
    description="AI-powered job scraping, scoring and resume tailoring",
    lifespan=lifespan,
)

container = create_container()
setup_dishka(
    container=container,
    app=app,
)


# CORS middleware - origins configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.effective_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(resumes_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(enrichment_router, prefix="/api/v1")
app.include_router(applications_router, prefix="/api/v1")
app.include_router(resume_wizard_router, prefix="/api/v1")


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
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
    )


if __name__ == "__main__":
    main()
