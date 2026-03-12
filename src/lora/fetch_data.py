"""
fetch_data.py

Downloads the Databricks Dolly 15k dataset, optionally filters by category,
then deterministically shuffles + subsamples into:
  - data/train.jsonl
  - data/eval.jsonl

Each line is a JSON object:
  {"id": "...", "category": "...", "prompt": "...", "response": "..."}

Usage examples:
  python -m src.fetch_data --n_train 800 --n_eval 15 --seed 42
  python -m src.fetch_data --category summarization --n_train 600 --n_eval 15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional      # load libraries

from datasets import load_dataset


def build_prompt(instruction: str, context: Optional[str]) -> str:
    """
    Formats Dolly examples into a consistent instruction prompt.

    Dolly fields are typically:
      - instruction (str)
      - context (str or "")
      - response (str)
      - category (str)
    """
    instruction = (instruction or "").strip()
    context = (context or "").strip()

    if context:
        return f"Instruction:\n{instruction}\n\nContext:\n{context}\n\nResponse:"
    return f"Instruction:\n{instruction}\n\nResponse:"


def write_jsonl(ds, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data", help="Output folder for JSONL files.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for deterministic subsampling.")
    parser.add_argument("--n_train", type=int, default=800, help="Number of training examples to keep.")
    parser.add_argument("--n_eval", type=int, default=15, help="Number of held-out eval examples.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional: filter to a single Dolly category (e.g., summarization, closed_qa).",
    )
    parser.add_argument(
        "--list_categories",
        action="store_true",
        help="Print available categories and exit.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # 1) Load dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    # 2) List categories if requested
    if args.list_categories:
        cats = sorted(set(ds["category"]))
        print("Available categories:")
        for c in cats:
            print(" -", c)
        return

    # 3) Optional filter by category
    if args.category is not None:
        target = args.category.strip()
        ds = ds.filter(lambda x: x["category"] == target)
        if len(ds) < args.n_train + args.n_eval:
            raise ValueError(
                f"Not enough examples in category='{target}'. "
                f"Have {len(ds)}, need at least {args.n_train + args.n_eval}."
            )

    # 4) Deterministic shuffle
    ds = ds.shuffle(seed=args.seed)

    # 5) Split eval first (prevents leakage)
    eval_raw = ds.select(range(args.n_eval))
    train_raw = ds.select(range(args.n_eval, args.n_eval + args.n_train))

    # 6) Convert to our unified JSONL schema
    def convert(example, split_name: str):
        prompt = build_prompt(example.get("instruction", ""), example.get("context", ""))
        response = (example.get("response", "") or "").strip()
        cat = (example.get("category", "") or "").strip()
        ex_id = example.get("id", None)
        if ex_id is None:
            # fallback deterministic-ish id if missing
            ex_id = f"{split_name}_{abs(hash(prompt)) % 10**12}"

        return {
            "id": str(ex_id),
            "category": cat,
            "prompt": prompt,
            "response": response,
        }

    train_out = [convert(ex, "train") for ex in train_raw]
    eval_out = [convert(ex, "eval") for ex in eval_raw]

    # 7) Write files
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    write_jsonl(train_out, train_path)
    write_jsonl(eval_out, eval_path)

    print(f"Wrote {len(train_out)} train examples to {train_path}")
    print(f"Wrote {len(eval_out)} eval examples to {eval_path}")

    # 8) Also write a small human-readable preview
    preview_path = out_dir / "preview.txt"
    with preview_path.open("w", encoding="utf-8") as f:
        f.write("=== EVAL PREVIEW (first 3) ===\n\n")
        for ex in eval_out[:3]:
            f.write(f"[{ex['id']}] category={ex['category']}\n")
            f.write(ex["prompt"] + "\n")
            f.write("----\n")
            f.write(ex["response"] + "\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"Wrote preview to {preview_path}")


if __name__ == "__main__":
    main()