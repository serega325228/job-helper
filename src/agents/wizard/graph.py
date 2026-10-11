from typing import Callable

from dishka.async_container import AsyncContainer
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

async def dishka_node(container: AsyncContainer, node_fn: Callable) -> Callable:
    async def wrapped(state):
        async with container() as c:
            return node_fn(c, state)
    return wrapped

def create_wizard_agent(
    models,
    tools,
) -> CompiledStateGraph:
    return create_agent(
        model=models.supervisor,
        tools=[
            tools.analyze_profile,
            tools.refresh_vacancies,
        ],
        system_prompt="SUPERVISOR_PROMPT", # Need to change
        checkpointer=
    )
