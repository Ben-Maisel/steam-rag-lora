"""
train_lora.py

LoRA fine-tunes a base Hugging Face causal LM on our train.jsonl.

Reads:
  data/train.jsonl  (each line: {"id","category","prompt","response"})

Writes:
  artifacts/lora_adapter/   (PEFT adapter weights + config)

Usage:
  python -m src.train_lora --model HuggingFaceTB/SmolLM2-360M-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import DataCollatorForSeq2Seq

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    # Keep prompt/response separate so we can mask loss on the prompt tokens.
    return Dataset.from_list(
        [{"id": r["id"], "prompt": r["prompt"], "response": r.get("response", "")} for r in rows]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/lora/steam_train.jsonl")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--out_dir", type=str, default="outputs/steam_lora_adapter")

    # Training hyperparams (small + safe defaults)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)

    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Precision
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training (can be unstable).")
    args = parser.parse_args()

    train_path = Path(args.train_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(train_path)
    if not rows:
        raise ValueError(f"No training rows found in {train_path}")

    print(f"Loading base model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model on a single device (avoid device_map='auto' sharding issues)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if (args.fp16 and device == "cuda") else torch.float32,
        device_map=None,
    ).to(device)

    # Needed for Trainer stability with generation models
    model.config.use_cache = False

    # LoRA config: SmolLM2 uses Llama-style projection module names
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    # Build dataset
    ds = build_dataset(rows)

    def tokenize_and_mask(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example["prompt"]
        response = example["response"]

        # Train to predict response given prompt.
        full_text = prompt + response

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        prompt_tok = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

        labels = full["input_ids"].copy()

        # Mask prompt tokens so loss is only on response portion
        prompt_len = len(prompt_tok["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        full["labels"] = labels
        return full

    ds_tok = ds.map(tokenize_and_mask, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer_ckpts"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=(args.fp16 and device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    print("Starting LoRA training...")
    trainer.train()

    # Save adapter weights
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()