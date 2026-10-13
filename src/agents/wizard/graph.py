from typing import Callable

from dishka.async_container import AsyncContainer
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from src.agents.wizard.state import WizardState
from src.agents.wizard.tools import WizardToolHandlers

async def dishka_node(container: AsyncContainer, node_fn: Callable) -> Callable:
    async def wrapped(state):
        async with container() as c:
            return node_fn(c, state)
    return wrapped

tools = WizardToolHandlers(resume_graph)

def create_wizard_agent(
    models,
    tools,
) -> CompiledStateGraph:
    return create_agent(
        model=models.wizard,
        tools=,
        system_prompt="WIZARD_PROMPT", # Need to change
        checkpointer=
    )
