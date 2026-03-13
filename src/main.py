from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import torch
import os





def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def print_device_info(device: str) -> None:
    print("\n" + "=" * 72)
    print("DEVICE INFORMATION".center(72))
    print("=" * 72)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Effective device:", device)

    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
        print("Visible GPUs:", visible)
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    print("=" * 72 + "\n", flush=True)


def print_banner() -> None:
    print("=" * 72, flush=True)
    print("Stitching Project End-to-End Pipeline".center(72), flush=True)
    print("=" * 72, flush=True)


def run(cmd: list[str], description: str, verbose: bool) -> None:
    print(f"\n{description} ...", flush=True)

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
    parser.add_argument("--verbose", action="store_true", help="Show subprocess logs.")
    parser.add_argument("--skip_train", action="store_true", help="Skip LoRA training.")
    parser.add_argument("--no_interactive", action="store_true", help="Do not start interactive mode at the end.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for LoRA training",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use for LoRA training",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = resolve_device(args.device)
    print_device_info(device)

    print_banner()

    # Paths used for existence checks
    reviews_csv = Path("data/steam_reviews_out/ALL_STEAM_REVIEWS.csv")
    embeddings_parquet = Path("data/embeddings/steam_review_chunks_with_embeddings.parquet")
    train_jsonl = Path("data/lora/steam_train.jsonl")
    eval_jsonl = Path("data/lora/steam_eval.jsonl")
    lora_adapter_dir = Path("outputs/steam_lora_adapter")

    # 1. Scrape Steam reviews
    if not reviews_csv.exists():
        run(
            [sys.executable, "data/scrape_steam_reviews.py"],
            "Scraping Steam reviews",
            args.verbose,
        )
    else:
        print("\nScraping Steam reviews ... skipped (reviews CSV already exists)", flush=True)

    # 2. Chunk + embed reviews
    if not embeddings_parquet.exists():
        run(
            [
                sys.executable,
                "src/chunk_and_embed.py",
                "--infile",
                str(reviews_csv),
                "--outdir",
                "data/embeddings",
            ],
            "Chunking and embedding Steam reviews",
            args.verbose,
        )
    else:
        print("\nChunking and embedding Steam reviews ... skipped (parquet already exists)", flush=True)

    # 3. Setup Pinecone env / index
    run(
        [sys.executable, "src/setup_pinecone_env.py"],
        "Setting up Pinecone environment",
        args.verbose,
    )

    # 4. Upload vectors to Pinecone
    run(
        [
            sys.executable,
            "src/upload_to_pinecone.py",
            "--parquet",
            str(embeddings_parquet),
            "--namespace",
            "steam-reviews",
            "--batch_size",
            "200",
        ],
        "Uploading vectors to Pinecone",
        args.verbose,
    )

    # 5. Build Steam LoRA train/eval data
    if not train_jsonl.exists() or not eval_jsonl.exists():
        run(
            [sys.executable, "-m", "src.lora.build_steam_lora_data"],
            "Building Steam LoRA dataset",
            args.verbose,
        )
    else:
        print("\nBuilding Steam LoRA dataset ... skipped (train/eval JSONL already exist)", flush=True)

    # 6. Train LoRA adapter
    if args.skip_train:
        print("\nTraining LoRA adapter ... skipped (--skip_train used)", flush=True)
    elif lora_adapter_dir.exists():
        print("\nTraining LoRA adapter ... skipped (adapter already exists)", flush=True)
    else:
        run(
            [
                sys.executable,
                "-m",
                "src.lora.train_lora",
                "--epochs",
                "3",
                "--device",
                device,
            ],
            "Training LoRA adapter",
            args.verbose,
        )

    # 7. Run the 6 evaluation questions
    run(
        [sys.executable, "-m", "src.agents.agentic_rag", "--mode", "test"],
        "Running 6-question system evaluation",
        args.verbose,
    )

    # 8. Evaluate latest response file with judge LLM
    run(
        [sys.executable, "-m", "src.evaluate_latest_responses"],
        "Evaluating latest responses with judge LLM",
        args.verbose,
    )

    # 9. Start interactive mode
    if not args.no_interactive:
        print("\nStarting interactive mode ...", flush=True)
        subprocess.run([sys.executable, "-m", "src.agents.agentic_rag", "--mode", "interactive"])
    else:
        print("\nInteractive mode ... skipped (--no_interactive used)", flush=True)

    print("\nPipeline complete.", flush=True)
    print("Key output folders:", flush=True)
    print(" - outputs/responses/", flush=True)
    print(" - outputs/evaluations/", flush=True)


if __name__ == "__main__":
    main()