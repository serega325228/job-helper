from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

def create_supervisor_agent(
    models,
    tools,
) -> CompiledStateGraph:
    return create_agent(
        model=models.supervisor,
        tools=[
            tools.analyze_profile,
            tools.refresh_vacancies,
        ],
        system_prompt=SUPERVISOR_PROMPT,
    )
