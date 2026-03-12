import os
import argparse
from datetime import datetime

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.agents.graph import build_graph
from src.agents.state import RAGState
from src.agents.nodes import (
    vectorstore,
    rebuild_docs_from_parquet,
)


load_dotenv()

TEST_QUESTIONS = [
    "What do people think of the graphics in Black Myth: Wukong?",
    "Is TEKKEN 8 balanced?",
    "What do people think of Blue Prince?",
    "What is the gameplay loop of Frostpunk 2 like?",
    "Is Kingdom Come: Deliverance 2 worth it?",
    "What do players think about the difficulty of Monster Hunter Wilds?",
]

PIPELINE_STEPS = [
    "rewrite_query",
    "retrieve_docs",
    "filter_relevant_docs",
    "generate_answer",
]


def build_initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "rewritten_question": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "answer": "",
        "route_decision": "",
    }


def run_question(app, question: str) -> str:
    result = app.invoke(build_initial_state(question))
    return result["answer"]


def build_basic_rag_prompt(question: str, docs) -> str:
    context_blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id", "unknown")
        game = meta.get("game_name", "unknown")
        sentiment = meta.get("sentiment", "unknown")

        context_blocks.append(
            f"[CHUNK {i}] chunk_id={chunk_id} game={game} sentiment={sentiment}\n{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)

    return (
        "You are answering questions using ONLY the provided Steam review chunks.\n"
        "If the chunks do not contain enough information to answer, say so clearly.\n\n"
        f"QUESTION: {question}\n\n"
        f"STEAM REVIEW CHUNKS:\n{context}\n\n"
        "ANSWER:"
    )


def run_base_llm(question: str) -> str:
    print("[System 1/4] Answering with base LLM (no RAG)...")
    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0.2)
    return llm.invoke(question).content.strip()


def run_basic_rag(question: str) -> str:
    print("[System 2/4] Answering with basic RAG...")
    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0.2)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(question)
    docs = rebuild_docs_from_parquet(docs)

    if not docs:
        return "No documents were retrieved for the question."

    prompt = build_basic_rag_prompt(question, docs)
    return llm.invoke(prompt).content.strip()


def run_advanced_agentic_rag(app, question: str, label: str) -> str:
    print(label)
    return run_question(app, question)


def run_advanced_agentic_rag_lora(app, question: str, label: str) -> str:
    print(label)
    return run_question(app, question)


def run_interactive_mode(app):
    while True:
        question = input("\nAsk a question (or type 'quit'): ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        if not question:
            continue

        answer = run_question(app, question)

        print("\nAdvanced agentic RAG answer:\n")
        print(answer)


def run_test_mode(base_app, lora_app):
    out_dir = os.path.join("outputs", "responses")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"assignment_outputs_{ts}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Stitching Project Evaluation Outputs\n")
        f.write(f"Timestamp: {ts}\n")
        f.write("Agent pipeline:\n")
        f.write(" -> ".join(PIPELINE_STEPS) + "\n")
        f.write("=" * 100 + "\n\n")

        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\nRunning question {i}/{len(TEST_QUESTIONS)}...")
            print(f"Q: {question}")

            base_llm_answer = run_base_llm(question)
            basic_rag_answer = run_basic_rag(question)
            advanced_base_answer = run_advanced_agentic_rag(
                base_app,
                question,
                "[System 3/4] Answering with advanced agentic RAG using OpenAI..."
            )
            advanced_lora_answer = run_advanced_agentic_rag_lora(
                lora_app,
                question,
                "[System 4/4] Answering with advanced agentic RAG using LoRA..."
            )

            print("Done.")

            f.write("=" * 100 + "\n")
            f.write(f"QUESTION {i}: {question}\n")
            f.write("=" * 100 + "\n\n")

            f.write("[1] BASE LLM (NO RAG)\n")
            f.write("-" * 100 + "\n")
            f.write(base_llm_answer + "\n\n")

            f.write("[2] BASIC RAG\n")
            f.write("-" * 100 + "\n")
            f.write(basic_rag_answer + "\n\n")

            f.write("[3] ADVANCED AGENTIC RAG WITH BASE MODEL\n")
            f.write("-" * 100 + "\n")
            f.write(advanced_base_answer + "\n\n")

            f.write("[4] ADVANCED AGENTIC RAG WITH FINE-TUNED MODEL\n")
            f.write("-" * 100 + "\n")
            f.write(advanced_lora_answer + "\n\n")

    print(f"\nFinished. Outputs saved to:\n{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["test", "interactive"],
        default="test",
        help="Run assignment test questions or interactive question loop.",
    )
    args = parser.parse_args()

    base_app = build_graph(answer_mode="base")
    lora_app = build_graph(answer_mode="lora")

    if args.mode == "interactive":
        run_interactive_mode(base_app)
    else:
        run_test_mode(base_app, lora_app)


if __name__ == "__main__":
    main()