from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.resume.nodes import (
    ask_user,
    choose_creation_method,
    prepare_question,
    save_resume,
)
from src.agents.resume.state import MasterResumeContext, MasterResumeState


def create_master_resume_agent(
    settings,
    models,
    tools,
    checkpointer: AsyncPostgresSaver,
) -> CompiledStateGraph:
    graph = StateGraph(
        MasterResumeState,
        context_schema=MasterResumeContext,
    )

    graph.add_node("choose_creation_method", choose_creation_method)
    graph.add_node("prepare_question", prepare_question)
    graph.add_node("ask_user", ask_user)
    graph.add_node("save_resume", save_resume)
    graph.add_node("create_based_on_profile", create_based_on_profile)

    graph.add_edge(START, "choose_creation_method")
    graph.add_conditional_edges(
        "choose_creation_method",
        lambda state: (
            "yes"
            if state.create_resume_manually.strip().lower() in ("yes", "y", "yeah")
            else "no"
        ),
        {
            "yes": "prepare_question",
            "no": "create_based_on_profile",
        },
    )
    graph.add_edge("prepare_question", "ask_user")
    graph.add_conditional_edges(
        "ask_user",
        lambda state: "save_resume" if state.is_complete else "ask_user",
        {
            "save_resume": "save_resume",
            "ask_user": "ask_user",
        },
    )
    graph.add_edge("save", END)
    graph.add_edge("create_based_on_profile", END)

    return graph.compile(checkpointer=checkpointer)
