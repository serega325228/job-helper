from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.profile.nodes import (
    ask_user,
    preferences_review,
    prepare_question,
    profile_review,
    save_profile,
    start,
)
from src.agents.profile.state import ProfileContext, ProfileState


def create_profile_agent(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph[ProfileState, ProfileContext, ProfileState, ProfileState]:
    graph = StateGraph(ProfileState, context_schema=ProfileContext)
    graph.add_node("start", start)
    graph.add_node("prepare_question", prepare_question)
    graph.add_node("ask_user", ask_user)
    graph.add_node("profile_review", profile_review)
    graph.add_node("preferences_review", preferences_review)
    graph.add_node("save_profile", save_profile)

    graph.add_edge(START, "start")
    graph.add_edge("start", "prepare_question")
    graph.add_conditional_edges(
        "prepare_question",
        lambda state: "ask_user" if state.missing_fields else "profile_review",
        {"ask_user": "ask_user", "profile_review": "profile_review"},
    )
    graph.add_edge("ask_user", "prepare_question")
    graph.add_conditional_edges(
        "profile_review",
        lambda state: (
            "preferences_review" if state.approved and state.preferences
            else "save_profile" if state.approved else "prepare_question"
        ),
        {"preferences_review": "preferences_review", "save_profile": "save_profile", "prepare_question": "prepare_question"},
    )
    graph.add_edge("preferences_review", "save_profile")
    graph.add_edge("save_profile", END)
    return graph.compile(checkpointer=checkpointer)
