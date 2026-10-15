from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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


def create_resume_agent(
    # settings,
    # models,
    # tools,
    checkpointer: AsyncPostgresSaver,
) -> CompiledStateGraph:
    graph = StateGraph(
        MasterResumeState,
        context_schema=MasterResumeContext,
    )

    graph.add_node("choose_creation_method", choose_creation_method)
    graph.add_node("manually_prepare_question", manually_prepare_question)
    graph.add_node("ask_user", ask_user)
    graph.add_node("save_resume", save_resume)
    graph.add_node("based_on_profile_prepare_question", based_on_profile_prepare_question)

    graph.add_edge(START, "choose_creation_method")
    graph.add_conditional_edges(
        "choose_creation_method",
        lambda state: (
            "yes"
            if state.create_resume_manually.strip().lower() in ("yes", "y", "yeah")
            else "no"
        ),
        {
            "yes": "manually_prepare_question",
            "no": "based_on_profile_prepare_question",
        },
    )
    graph.add_edge("manually_prepare_question", "ask_user")
    graph.add_conditional_edges(
        "ask_user",
        lambda state: "save_resume" if state.is_complete else "keep_asking",
        {
            "save_resume": "save_resume",
            "keep_asking": "manually_prepare_question",
        },
    )


    graph.add_edge("based_on_profile_prepare_question", "ask_user")

    # Current version
    graph.add_conditional_edges(
        "ask_user",
        lambda state: "keep_asking" if len(state.missing_fields) > 0 else "save_resume",
        {
            "save_resume": "save_resume",
            "keep_asking": "based_on_profile_prepare_question",
        },
    )
    graph.add_edge("save_resume", END)

    return graph.compile(checkpointer=checkpointer)
