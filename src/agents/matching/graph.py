from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.agents.matching.nodes import (
    compare_candidates,
    rerank_candidates,
    search_candidates,
)
from src.agents.matching.state import MatchingContext, MatchingState


def create_matching_graph() -> CompiledStateGraph:
    graph = StateGraph(MatchingState, context_schema=MatchingContext)

    graph.add_node("search_candidates", search_candidates)
    graph.add_node("compare_candidates", compare_candidates)
    graph.add_node("rerank_candidates", rerank_candidates)

    graph.add_edge(START, "search_candidates")
    graph.add_edge("search_candidates", "compare_candidates")
    graph.add_edge("compare_candidates", "rerank_candidates")
    graph.add_edge("rerank_candidates", END)

    return graph.compile()


create_match_workflow = create_matching_graph
