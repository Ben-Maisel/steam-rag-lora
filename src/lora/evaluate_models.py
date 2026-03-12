"""
evaluate_models.py

Compares base model outputs vs LoRA outputs against references.

Reads:
  artifacts/base_outputs.jsonl
  artifacts/lora_outputs_3ep.jsonl

Writes:
  artifacts/eval_metrics.json
  artifacts/eval_summary.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from rouge_score import rouge_scorer


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def rougeL_f1(scorer, ref: str, pred: str) -> float:
    return scorer.score(ref, pred)["rougeL"].fmeasure

def compute_metrics_and_per_example(rows: List[Dict]) -> Dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_vals = []
    exact_matches = 0

    for r in rows:
        ref = r["reference_response"]
        pred = r["model_output"]
        rouge_vals.append(rougeL_f1(scorer, ref, pred))
        if pred.strip() == ref.strip():
            exact_matches += 1

    n = len(rows)
    return {
        "rougeL_f1_avg": sum(rouge_vals) / n,
        "exact_match_rate": exact_matches / n,
        "rougeL_f1_per_example": rouge_vals,
    }

def main() -> None:
    base_path = Path("artifacts/base_outputs.jsonl")
    lora_path = Path("artifacts/lora_outputs_3ep.jsonl")

    base_rows = read_jsonl(base_path)
    lora_rows = read_jsonl(lora_path)

    # sanity: align by id
    base_by_id = {r["id"]: r for r in base_rows}
    lora_by_id = {r["id"]: r for r in lora_rows}
    ids = [i for i in base_by_id.keys() if i in lora_by_id]

    base_aligned = [base_by_id[i] for i in ids]
    lora_aligned = [lora_by_id[i] for i in ids]

    base_eval = compute_metrics_and_per_example(base_aligned)
    lora_eval = compute_metrics_and_per_example(lora_aligned)

    # find best improvement example
    diffs = [
        (ids[i], lora_eval["rougeL_f1_per_example"][i] - base_eval["rougeL_f1_per_example"][i])
        for i in range(len(ids))
    ]
    best_id, best_gain = max(diffs, key=lambda x: x[1])
    sample_base = base_by_id[best_id]
    sample_lora = lora_by_id[best_id]

    results = {
        "base_model": {k: v for k, v in base_eval.items() if k != "rougeL_f1_per_example"},
        "lora_model": {k: v for k, v in lora_eval.items() if k != "rougeL_f1_per_example"},
        "best_example": {"id": best_id, "rougeL_gain": best_gain},
    }

    metrics_path = Path("artifacts/eval_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))

    summary_path = Path("artifacts/eval_summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== Evaluation Summary ===\n\n")
        f.write("BASE MODEL METRICS:\n")
        f.write(json.dumps(results["base_model"], indent=2))
        f.write("\n\n")

        f.write("LORA MODEL METRICS:\n")
        f.write(json.dumps(results["lora_model"], indent=2))
        f.write("\n\n")

        f.write("BEST IMPROVEMENT EXAMPLE:\n")
        f.write(json.dumps(results["best_example"], indent=2))
        f.write("\n\n")

        f.write("=== Sample Comparison (Best ROUGE-L Gain) ===\n\n")
        f.write("PROMPT:\n")
        f.write(sample_base["prompt"] + "\n\n")

        f.write("BASE OUTPUT:\n")
        f.write(sample_base["model_output"] + "\n\n")

        f.write("LORA OUTPUT:\n")
        f.write(sample_lora["model_output"] + "\n\n")

        f.write("REFERENCE:\n")
        f.write(sample_base["reference_response"] + "\n")

    print("Saved evaluation results to artifacts/eval_metrics.json")
    print("Saved readable summary to artifacts/eval_summary.txt")


if __name__ == "__main__":
    main()