import os
from langgraph.graph import StateGraph, END

from src.agents.state import RAGState
from src.agents.nodes import (
    rewrite_query,
    retrieve_docs,
    filter_relevant_docs,
    decide_next_step,
    generate_answer_base,
    generate_answer_lora,
)


def build_graph(answer_mode: str = "base"):
    graph = StateGraph(RAGState)

    if answer_mode == "lora":
        answer_node_fn = generate_answer_lora
    else:
        answer_node_fn = generate_answer_base

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("filter_relevant_docs", filter_relevant_docs)
    graph.add_node("decide_next_step", decide_next_step)
    graph.add_node("generate_answer", answer_node_fn)

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
            output_file_path=f"outputs/agent_graph_{answer_mode}.png",
        )
    except Exception:
        pass

    return app