from langgraph.graph.state import CompiledStateGraph
from pydantic.dataclasses import dataclass

from src.agents.supervisor.tools import create_supervisor_tools
from src.config.settings import Settings
from src.graphs.supervisor.tools import SupervisorToolHandlers


@dataclass(slots=True)
class Application:
    supervisor: CompiledStateGraph
    services: ServiceContainer
    repositores: RepositoryContainer

def create_application() -> Application:
    settings = Settings()
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

    return Application(supervisor, services, repositories)
