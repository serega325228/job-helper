from agents.collection.nodes import collect_raw_vacancies, normalize_vacancies, save_vacancies
from agents.collection.state import CollectionGraphContext, CollectionState
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph


def create_search_workflow(
    models,
    tools
) -> CompiledStateGraph:
    graph = StateGraph(
        CollectionState,
        context_schema=CollectionGraphContext
    )

    graph.add_node("collect", collect_raw_vacancies)
    graph.add_node("normalize", normalize_vacancies)
    graph.add_node("save", save_vacancies)

    graph.add_edge(START, "collect")
    graph.add_edge("collect", "normalize")
    graph.add_edge("normalize", "save")
    graph.add_edge("save", END)

    return graph.compile()
