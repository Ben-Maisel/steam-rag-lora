"""
run_base_model.py

Loads a base Hugging Face causal LM and generates outputs for the eval prompts.

Reads:
  data/eval.jsonl  (each line: {"id","category","prompt","response"})

Writes:
  artifacts/base_outputs.jsonl
  artifacts/base_preview.txt

Usage:
  python -m src.run_base_model --model HuggingFaceTB/SmolLM2-360M-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List
from peft import PeftModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, default="data/eval.jsonl")
    parser.add_argument("--out_path", type=str, default="artifacts/base_outputs.jsonl")
    parser.add_argument("--preview_path", type=str, default="artifacts/base_preview.txt")

    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Base model name on Hugging Face.",
    )
    
    parser.add_argument(
    "--lora_path",
    type=str,
    default="",
    help="Optional path to a PEFT LoRA adapter directory (e.g., artifacts/lora_adapter).",
    
    )

    # Generation controls (keep fixed for fair base vs finetuned comparison)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--min_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (off by default).")

    # Logging / verbosity
    parser.add_argument("--verbose", action="store_true", help="Show HF warnings/logs.")

    # Device
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # ---- Reduce noise from HF + Python warnings (especially in Docker/Windows) ----
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not args.verbose:
        warnings.filterwarnings("ignore", category=UserWarning)
        hf_logging.set_verbosity_error()

    eval_path = Path(args.eval_path)
    out_path = Path(args.out_path)
    preview_path = Path(args.preview_path)

    examples = read_jsonl(eval_path)
    if len(examples) == 0:
        raise ValueError(f"No examples found in {eval_path}")

    # Device selection
    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
        device_map=None,
    )
    if device == "cuda":
        model.to("cuda")
    elif device == "cpu":
        model.to("cpu")
    model.eval()

    if args.lora_path:
        print(f"Loading LoRA adapter from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.to(device)
        model.eval()

    outputs: List[Dict[str, Any]] = []
    
    for ex in examples:
        prompt = ex["prompt"]

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = gen[0][input_len:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        outputs.append(
            {
                "id": ex["id"],
                "category": ex.get("category", ""),
                "prompt": prompt,
                "reference_response": ex.get("response", ""),
                "model": args.model,
                "generation_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "min_new_tokens": args.min_new_tokens,
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                "model_output": pred_text,
            }
        )

    write_jsonl(outputs, out_path)
    print(f"Wrote base outputs to {out_path}")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8") as f:
        for ex in outputs:
            f.write(f"=== ID: {ex['id']} | category={ex['category']} ===\n\n")
            f.write("PROMPT:\n")
            f.write(ex["prompt"] + "\n\n")
            f.write("BASE MODEL OUTPUT:\n")
            f.write(ex["model_output"] + "\n\n")
            f.write("REFERENCE (DATASET ANSWER):\n")
            f.write(ex["reference_response"] + "\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"Wrote preview to {preview_path}")


if __name__ == "__main__":
    main()