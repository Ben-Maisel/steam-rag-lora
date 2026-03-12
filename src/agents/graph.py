import os
from langgraph.graph import StateGraph, END

from src.agents.state import RAGState
from src.agents.nodes import (
    rewrite_query,
    retrieve_docs,
    filter_relevant_docs,
    decide_next_step,
    generate_answer,
)


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("filter_relevant_docs", filter_relevant_docs)
    graph.add_node("decide_next_step", decide_next_step)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("rewrite_query")

    graph.add_edge("rewrite_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "filter_relevant_docs")
    graph.add_edge("filter_relevant_docs", "decide_next_step")

    graph.add_conditional_edges(
        "decide_next_step",
        lambda state: state["route_decision"],
        {
            "answer": "generate_answer",
            "retrieve": "retrieve_docs",
        },
    )

    graph.add_edge("generate_answer", END)

    app = graph.compile()

    os.makedirs("outputs", exist_ok=True)

    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            output_file_path="outputs/agent_graph.png",
        )
    except Exception:
        pass

    return app