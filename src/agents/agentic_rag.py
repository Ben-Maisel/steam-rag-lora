import os
from datetime import datetime

from src.agents.graph import build_graph
from src.agents.state import RAGState
import argparse


TEST_QUESTIONS = [
    "What do people think of the graphics in Black Myth: Wukong?",
    "Is TEKKEN 8 balanced?",
    "What do people think of Blue Prince?",
    "What is the gameplay loop of Frostpunk 2 like?",
    "Is Kingdom Come: Deliverance 2 worth it?",
    "What do players think about the difficulty of Monster Hunter Wilds?"
]

PIPELINE_STEPS = [
    "rewrite_query",
    "retrieve_docs",
    "filter_relevant_docs",
    "generate_answer",
]


def run_question(app, question: str) -> str:
    initial_state: RAGState = {
        "question": question,
        "rewritten_question": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "answer": "",
        "route_decision": "",
    }

    result = app.invoke(initial_state)
    return result["answer"]

def run_interactive_mode(app):
    while True:
        question = input("\nAsk a question (or type 'quit'): ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        if not question:
            continue

        answer = run_question(app, question)

        print("\nAnswer:\n")
        print(answer)

def run_test_mode(app):
    os.makedirs("outputs", exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"agentic_rag_outputs_{ts}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Agentic RAG Test Outputs\n")
        f.write(f"Timestamp: {ts}\n")
        f.write("Agent pipeline:\n")
        f.write(" -> ".join(PIPELINE_STEPS) + "\n")
        f.write("=" * 100 + "\n\n")

        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\nRunning question {i}/{len(TEST_QUESTIONS)}...")
            print(f"Q: {question}")

            answer = run_question(app, question)

            print("Done.")

            f.write("=" * 100 + "\n")
            f.write(f"QUESTION {i}: {question}\n")
            f.write("-" * 100 + "\n")
            f.write(f"ANSWER:\n{answer}\n\n")

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

    app = build_graph()

    if args.mode == "interactive":
        run_interactive_mode(app)
    else:
        run_test_mode(app)


if __name__ == "__main__":
    main()