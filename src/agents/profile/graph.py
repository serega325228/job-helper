from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.profile.nodes import (
    ask_user,
    prepare_question,
    profile_review,
    save_profile,
    start,
)
from src.agents.profile.state import ProfileContext, ProfileState


def create_profile_agent(
    # settings,
    # models,
    # tools,
    checkpointer: AsyncPostgresSaver,
) -> CompiledStateGraph:
    graph = StateGraph(
        ProfileState,
        context_schema=ProfileContext,
    )

    graph.add_node("start", start)
    graph.add_node("prepare_question", prepare_question)
    graph.add_node("ask_user", ask_user)
    graph.add_node("profile_review", profile_review)
    graph.add_node("save_profile", save_profile)

    graph.add_edge(START, "start")
    graph.add_conditional_edges(
        "start",
        lambda state: (
            "prepare_question" if state.profile_exists else "end"
        ),
        {
            "prepare_question": "prepare_question",
            "end": END,
        },
    )
    graph.add_edge("prepare_question", "ask_user")
    graph.add_conditional_edges(
        "ask_user",
        lambda state: (
            "keep_asking" if len(state.missing_fields) > 0 else "review_profile"
        ),
        {
            "keep_asking": "prepare_question",
            "review_profile": "profile_review",
        },
    )
    graph.add_edge("profile_review", "save_profile")
    graph.add_edge("save_profile", END)

    return graph.compile(checkpointer=checkpointer)
