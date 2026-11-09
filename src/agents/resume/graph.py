from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.resume.nodes import (
    ask_user,
    based_on_profile_prepare_question,
    choose_creation_method,
    manually_prepare_question,
    save_resume,
)
from src.agents.resume.state import MasterResumeContext, MasterResumeState


def preparation_node(state: MasterResumeState) -> str:
    if state.creation_method == "manual":
        return "manually_prepare_question"
    if state.creation_method == "profile":
        return "based_on_profile_prepare_question"
    raise ValueError("A resume creation method must be selected before analysis")


def create_resume_agent(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph[
    MasterResumeState, MasterResumeContext, MasterResumeState, MasterResumeState
]:
    graph = StateGraph(MasterResumeState, context_schema=MasterResumeContext)
    graph.add_node("choose_creation_method", choose_creation_method)
    graph.add_node("manually_prepare_question", manually_prepare_question)
    graph.add_node("based_on_profile_prepare_question", based_on_profile_prepare_question)
    graph.add_node("ask_user", ask_user)
    graph.add_node("save_resume", save_resume)

    prepare_routes = ["manually_prepare_question", "based_on_profile_prepare_question"]
    graph.add_edge(START, "choose_creation_method")
    graph.add_conditional_edges("choose_creation_method", preparation_node, prepare_routes)
    for node in prepare_routes:
        graph.add_conditional_edges(
            node,
            lambda state: "ask_user" if state.missing_fields else "save_resume",
            {"ask_user": "ask_user", "save_resume": "save_resume"},
        )
    # Every answer is incorporated before the next completion decision.
    graph.add_conditional_edges("ask_user", preparation_node, prepare_routes)
    graph.add_edge("save_resume", END)
    return graph.compile(checkpointer=checkpointer)
