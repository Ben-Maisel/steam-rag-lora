"""
pipeline.py

End-to-end pipeline for Assignment 3:

1) fetch data
2) train LoRA
3) run base model
4) run LoRA model
5) evaluate
6) produce artifacts/eval_metrics.json and artifacts/eval_summary.txt

Usage:
  python -m src.pipeline --model HuggingFaceTB/SmolLM2-360M-Instruct --epochs 3 --device auto
  python -m src.pipeline --verbose   # show subprocess logs
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import torch


def resolve_device(device_arg: str) -> str:
    """
    Resolve the effective device to use.

    - auto: use CUDA if available else CPU
    - cuda/cpu: use exactly what the user specified
    """
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def print_device_info(effective_device: str) -> None:
    """
    Print torch + device info so graders/users can clearly see whether
    the run is using GPU or CPU.
    """
    print("=" * 60)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Effective device:", effective_device)

    if effective_device == "cuda":
        # Safe because effective_device == "cuda" implies is_available() True in resolve_device(auto).
        # If user forced --device cuda on a machine without CUDA, this may raise — which is fine.
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    print("=" * 60, flush=True)


def run(cmd: list[str], description: str, verbose: bool) -> None:
    print(f"{description} ...", flush=True)

    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if result.returncode != 0:
        raise RuntimeError(f"{description} failed.")
    print(f"{description} ... done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--verbose", action="store_true", help="Show subprocess logs.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print_device_info(device)

    # 1) fetch data
    # NOTE: assumes you have src/fetch_data.py. If your fetch script name differs, change it here.
    run(
    [
        sys.executable,
        "-m",
        "src.fetch_data",
        "--category",
        "summarization",
        "--n_train",
        "500",
        "--n_eval",
        "15",
        "--seed",
        "42",
    ],
    "Fetching data",
    args.verbose,
    )

    # 2) train LoRA
    # NOTE: If your train_lora module supports a --device argument, you can add:
    # "--device", device
    # to the command list below.
    run(
        [
            sys.executable,
            "-m",
            "src.train_lora",
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--grad_accum",
            str(args.grad_accum),
            "--max_length",
            str(args.max_length),
        ],
        "Training LoRA",
        args.verbose,
    )

    # 3) run base model
    run(
        [
            sys.executable,
            "-m",
            "src.run_base_model",
            "--model",
            args.model,
            "--out_path",
            "artifacts/base_outputs.jsonl",
            "--preview_path",
            "artifacts/base_preview.txt",
            "--device",
            device,
        ],
        "Running base model",
        args.verbose,
    )

    # 4) run LoRA model
    run(
        [
            sys.executable,
            "-m",
            "src.run_base_model",
            "--model",
            args.model,
            "--lora_path",
            "artifacts/lora_adapter",
            "--out_path",
            "artifacts/lora_outputs_3ep.jsonl",
            "--preview_path",
            "artifacts/lora_preview_3ep.txt",
            "--device",
            device,
        ],
        "Running LoRA model",
        args.verbose,
    )

    # 5) evaluate
    run([sys.executable, "-m", "src.evaluate_models"], "Evaluating models", args.verbose)

    # 6) done
    print("\nPipeline complete. Key outputs:", flush=True)
    print(" - artifacts/eval_metrics.json", flush=True)
    print(" - artifacts/eval_summary.txt", flush=True)


if __name__ == "__main__":
    main()