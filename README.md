# Steam Game Review RAG Agent

An agentic retrieval-augmented generation (RAG) system built over Steam game reviews, featuring LoRA fine-tuning and automated evaluation across four system configurations.

---

## Overview

This project builds and benchmarks four NLP systems for answering questions about video games using real Steam user reviews as a knowledge base:

1. **Base LLM** — zero-shot responses with no retrieval
2. **Basic RAG** — standard retrieval over embedded review chunks
3. **Agentic RAG** — LangGraph-based agent with tool-calling and multi-step reasoning
4. **Agentic RAG + LoRA** — same agent with a LoRA fine-tuned model for domain adaptation

An automated LLM judge scores each system on groundedness, completeness, nuance, and clarity to produce a ranked comparison.

---

## Architecture

- **Data ingestion**: Steam reviews scraped, chunked, and embedded
- **Vector store**: Pinecone for semantic similarity search
- **Agent framework**: LangGraph for multi-step agentic retrieval
- **Fine-tuning**: LoRA adapter trained on a review-based QA dataset via HuggingFace PEFT
- **Evaluation**: LLM-as-judge scoring across 5 dimensions per response

---

## Quickstart

```bash
git clone https://github.com/Ben-Maisel/steam-rag-lora.git
cd steam-rag-lora

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

cp .env.example .env        # Add your Pinecone and OpenAI API keys

# Install PyTorch — GPU (CUDA 12.1) recommended
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Or CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python -m src.main --gpu 0   # change --gpu to your device index; required
```

---

## Project Structure

```
├── src/
│   ├── main.py                  # Pipeline entry point
│   ├── scraper/                 # Steam review ingestion
│   ├── embeddings/              # Chunking and embedding pipeline
│   ├── vectorstore/             # Pinecone index management
│   ├── rag/                     # Basic and agentic RAG implementations
│   ├── finetuning/              # LoRA fine-tuning pipeline
│   └── evaluation/              # LLM judge and scoring
├── outputs/
│   ├── agent_graph_base.png     # LangGraph agent graph (base model)
│   ├── agent_graph_lora.png     # LangGraph agent graph (LoRA model)
│   ├── steam_lora_eval_outputs.jsonl
│   ├── steam_lora_eval_preview.txt
│   ├── responses/               # System responses to evaluation questions
│   ├── evaluations/             # LLM judge scores and rankings
│   └── steam_lora_adapter/      # Trained LoRA adapter weights
├── .env.example
└── requirements.txt
```

---

## Pre-generated Outputs

Outputs are included for quick inspection without running the pipeline:

- `outputs/responses/` — Responses from all 4 systems across evaluation questions
- `outputs/evaluations/` — JSON + human-readable evaluation reports with per-question scores
- `outputs/agent_graph_*.png` — LangGraph agent graph visualizations
- `outputs/steam_lora_eval_preview.txt` — Side-by-side system comparison with commentary

---

## Configuration

Set the following in `.env`:

```
PINECONE_API_KEY=your_key
OPENAI_API_KEY=your_key
```

---

## Tech Stack

Python · LangGraph · HuggingFace Transformers · PEFT (LoRA) · Pinecone · OpenAI · PyTorch

---

## License

MIT
