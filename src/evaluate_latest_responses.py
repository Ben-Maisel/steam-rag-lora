import os
import re
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_KEYS = {
    "1": "base_llm_no_rag",
    "2": "basic_rag",
    "3": "advanced_agentic_rag_base_model",
    "4": "advanced_agentic_rag_fine_tuned_model",
}

SYSTEM_LABELS = {
    "base_llm_no_rag": "Base LLM (No RAG)",
    "basic_rag": "Basic RAG",
    "advanced_agentic_rag_base_model": "Advanced Agentic RAG with Base Model",
    "advanced_agentic_rag_fine_tuned_model": "Advanced Agentic RAG with Fine-Tuned Model",
}


def find_latest_response_file(responses_dir: str) -> Path:
    files = glob.glob(os.path.join(responses_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {responses_dir}")
    latest = max(files, key=os.path.getmtime)
    return Path(latest)


def parse_response_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    question_pattern = re.compile(
        r"QUESTION\s+(\d+):\s*(.*?)\n=+\n\n"
        r"\[1\]\s*BASE LLM \(NO RAG\)\n-+\n(.*?)\n\n"
        r"\[2\]\s*BASIC RAG\n-+\n(.*?)\n\n"
        r"\[3\]\s*ADVANCED AGENTIC RAG WITH BASE MODEL\n-+\n(.*?)\n\n"
        r"\[4\]\s*ADVANCED AGENTIC RAG WITH FINE-TUNED MODEL\n-+\n(.*?)(?=\n=+\nQUESTION|\Z)",
        re.DOTALL,
    )

    parsed = []
    for match in question_pattern.finditer(text):
        q_num, question, s1, s2, s3, s4 = match.groups()
        parsed.append(
            {
                "question_number": int(q_num),
                "question": question.strip(),
                "responses": {
                    "base_llm_no_rag": s1.strip(),
                    "basic_rag": s2.strip(),
                    "advanced_agentic_rag_base_model": s3.strip(),
                    "advanced_agentic_rag_fine_tuned_model": s4.strip(),
                },
            }
        )

    if not parsed:
        raise ValueError("Could not parse any questions from the response file.")

    return parsed


def build_judge_prompt(question_block: Dict[str, Any]) -> str:
    q = question_block["question"]
    r = question_block["responses"]

    return f"""
You are evaluating outputs for a course project comparing 4 GenAI systems on Steam review question answering.

IMPORTANT CONTEXT ABOUT THE PROJECT
- The 4 compared systems are:
  1. Base LLM (no RAG)
  2. Basic RAG
  3. Advanced agentic RAG with base model
  4. Advanced agentic RAG with fine-tuned model
- The advanced agentic RAG pipeline uses multiple LLM-based agents such as query rewriting, retrieval, filtering, and answer generation.
- The fine-tuned model is a LoRA fine-tuned small Hugging Face model.
- The LoRA model was trained on a very small dataset (roughly 54 training examples and 9 eval examples), so it is expected to be at a major disadvantage compared with a very strong proprietary model like GPT-4o-mini/OpenAI-based responses.
- Therefore, do NOT judge the fine-tuned model by asking whether it beats a frontier model overall. Instead, judge whether it still gives a relevant, grounded, useful answer relative to the retrieved-review task.
- The goal of this evaluation is not to declare one model universally superior, but to compare quality across the 4 systems for this question in a fair, assignment-appropriate way.
- Base LLM has no retrieval, so it may be fluent but outdated, generic, or hallucinated.
- RAG systems should be judged heavily on grounding to the likely Steam-review evidence and on whether they address mixed sentiment when appropriate.

SCORING RUBRIC
For EACH of the 4 responses, score from 1 to 5 on:
1. groundedness
   - Is the answer plausibly grounded in review evidence rather than generic prior knowledge?
   - For no-RAG answers, low scores are appropriate if the answer is generic, outdated, or speculative.
2. completeness
   - Does it actually answer the question?
3. nuance
   - Does it mention both positives and negatives when appropriate, or otherwise reflect mixed sentiment well?
4. clarity
   - Is it readable, coherent, and well-structured?
5. overall_quality
   - Overall usefulness for answering the user's question in this project context.

SCORING GUIDANCE
- 5 = excellent
- 4 = strong
- 3 = acceptable / mixed
- 2 = weak
- 1 = poor / failed

PLEASE ALSO:
- Rank the 4 systems from best to worst for this question.
- Give a short rationale for each system.
- Give one short overall comparison paragraph.

IMPORTANT OUTPUT FORMAT
Return STRICT JSON only.
Use exactly this schema:

{{
  "question": "...",
  "scores": {{
    "base_llm_no_rag": {{
      "groundedness": 1,
      "completeness": 1,
      "nuance": 1,
      "clarity": 1,
      "overall_quality": 1,
      "rationale": "..."
    }},
    "basic_rag": {{
      "groundedness": 1,
      "completeness": 1,
      "nuance": 1,
      "clarity": 1,
      "overall_quality": 1,
      "rationale": "..."
    }},
    "advanced_agentic_rag_base_model": {{
      "groundedness": 1,
      "completeness": 1,
      "nuance": 1,
      "clarity": 1,
      "overall_quality": 1,
      "rationale": "..."
    }},
    "advanced_agentic_rag_fine_tuned_model": {{
      "groundedness": 1,
      "completeness": 1,
      "nuance": 1,
      "clarity": 1,
      "overall_quality": 1,
      "rationale": "..."
    }}
  }},
  "ranking_best_to_worst": [
    "advanced_agentic_rag_base_model",
    "basic_rag",
    "base_llm_no_rag",
    "advanced_agentic_rag_fine_tuned_model"
  ],
  "overall_comparison": "..."
}}

QUESTION
{q}

RESPONSE 1 - Base LLM (No RAG)
{r["base_llm_no_rag"]}

RESPONSE 2 - Basic RAG
{r["basic_rag"]}

RESPONSE 3 - Advanced Agentic RAG with Base Model
{r["advanced_agentic_rag_base_model"]}

RESPONSE 4 - Advanced Agentic RAG with Fine-Tuned Model
{r["advanced_agentic_rag_fine_tuned_model"]}
""".strip()


def judge_question(
    client: OpenAI,
    model: str,
    question_block: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_judge_prompt(question_block)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Judge model did not return valid JSON for question "
            f"{question_block['question_number']}.\nRaw output:\n{text}"
        ) from e


def compute_summary(judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = ["groundedness", "completeness", "nuance", "clarity", "overall_quality"]

    totals: Dict[str, Dict[str, float]] = {
        k: {m: 0.0 for m in metrics} for k in SYSTEM_LABELS.keys()
    }

    for item in judgments:
        for system_key, score_block in item["scores"].items():
            for metric in metrics:
                totals[system_key][metric] += score_block[metric]

    n = len(judgments)
    averages = {
        system_key: {metric: round(totals[system_key][metric] / n, 3) for metric in metrics}
        for system_key in totals
    }

    overall_ranking = sorted(
        averages.items(),
        key=lambda x: x[1]["overall_quality"],
        reverse=True,
    )

    return {
        "num_questions_evaluated": n,
        "average_scores": averages,
        "ranking_by_average_overall_quality": [
            {"system_key": k, "label": SYSTEM_LABELS[k], "overall_quality": v["overall_quality"]}
            for k, v in overall_ranking
        ],
    }


def write_human_readable_report(
    out_path: Path,
    source_file: Path,
    judge_model: str,
    judgments: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    lines = []
    lines.append("LLM Evaluation of Latest Stitching Project Responses")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source response file: {source_file}")
    lines.append(f"Judge model: {judge_model}")
    lines.append("=" * 100)
    lines.append("")

    lines.append("Average Scores")
    lines.append("-" * 100)
    for system_key, avg_block in summary["average_scores"].items():
        lines.append(f"{SYSTEM_LABELS[system_key]}")
        for metric, value in avg_block.items():
            lines.append(f"  {metric}: {value}")
        lines.append("")

    lines.append("Ranking by Average Overall Quality")
    lines.append("-" * 100)
    for i, item in enumerate(summary["ranking_by_average_overall_quality"], 1):
        lines.append(f"{i}. {item['label']} - overall_quality={item['overall_quality']}")
    lines.append("")

    for item in judgments:
        lines.append("=" * 100)
        lines.append(f"QUESTION: {item['question']}")
        lines.append("=" * 100)
        lines.append("")

        for system_key in SYSTEM_LABELS.keys():
            block = item["scores"][system_key]
            lines.append(SYSTEM_LABELS[system_key])
            lines.append("-" * 100)
            lines.append(
                f"groundedness={block['groundedness']}, "
                f"completeness={block['completeness']}, "
                f"nuance={block['nuance']}, "
                f"clarity={block['clarity']}, "
                f"overall_quality={block['overall_quality']}"
            )
            lines.append(f"rationale: {block['rationale']}")
            lines.append("")

        lines.append("Best to worst ranking:")
        for rank, system_key in enumerate(item["ranking_best_to_worst"], 1):
            lines.append(f"{rank}. {SYSTEM_LABELS.get(system_key, system_key)}")
        lines.append("")
        lines.append("Overall comparison:")
        lines.append(item["overall_comparison"])
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_dir", type=str, default="outputs/responses")
    parser.add_argument("--evaluations_dir", type=str, default="outputs/evaluations")
    parser.add_argument("--judge_model", type=str, default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max_questions", type=int, default=0, help="If >0, only judge the first N questions.")
    args = parser.parse_args()

    load_dotenv()

    latest_file = find_latest_response_file(args.responses_dir)
    parsed = parse_response_file(latest_file)

    if args.max_questions > 0:
        parsed = parsed[: args.max_questions]

    client = OpenAI()

    judgments = []
    for item in parsed:
        print(f"Evaluating question {item['question_number']}/{len(parsed)}...")
        judged = judge_question(client, args.judge_model, item)
        judgments.append(judged)

    summary = compute_summary(judgments)

    out_dir = Path(args.evaluations_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = out_dir / f"evaluation_{ts}.json"
    txt_out = out_dir / f"evaluation_{ts}.txt"

    payload = {
        "source_response_file": str(latest_file),
        "judge_model": args.judge_model,
        "summary": summary,
        "judgments": judgments,
    }

    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_human_readable_report(txt_out, latest_file, args.judge_model, judgments, summary)

    print(f"Latest response file: {latest_file}")
    print(f"Saved JSON evaluation to: {json_out}")
    print(f"Saved text evaluation to: {txt_out}")


if __name__ == "__main__":
    main()