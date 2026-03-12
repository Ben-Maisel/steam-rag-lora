import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.agents.state import RAGState


load_dotenv()

chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(model=chat_model, temperature=0.2)
embeddings = OpenAIEmbeddings(model=embed_model)

parquet_path = os.getenv(
    "EMBEDDINGS_PARQUET",
    "data/embeddings/steam_review_chunks_with_embeddings.parquet"
)

chunks_df = pd.read_parquet(parquet_path).set_index("chunk_id")

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "steam-reviews")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "steam-reviews")

vectorstore = PineconeVectorStore(
    index_name=pinecone_index_name,
    embedding=embeddings,
    namespace=pinecone_namespace,
    text_key="chunk_id",
)

# -----------------------------
# LoRA answer model setup
# -----------------------------
ANSWER_MODEL_NAME = os.getenv("ANSWER_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "outputs/steam_lora_adapter")
ANSWER_DEVICE = os.getenv("ANSWER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_answer_tokenizer = None
_answer_model = None


def get_answer_model():
    global _answer_tokenizer, _answer_model

    if _answer_tokenizer is not None and _answer_model is not None:
        return _answer_tokenizer, _answer_model

    print(f"[Agent] Loading answer model: {ANSWER_MODEL_NAME}")
    print(f"[Agent] Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    print(f"[Agent] Using answer device: {ANSWER_DEVICE}")

    _answer_tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL_NAME, use_fast=True)
    _answer_tokenizer.padding_side = "left"
    if _answer_tokenizer.pad_token is None:
        _answer_tokenizer.pad_token = _answer_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        ANSWER_MODEL_NAME,
        torch_dtype=torch.float16 if ANSWER_DEVICE == "cuda" else torch.float32,
        device_map=None,
    )
    base_model.to(ANSWER_DEVICE)
    base_model.eval()

    _answer_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    _answer_model.to(ANSWER_DEVICE)
    _answer_model.eval()

    return _answer_tokenizer, _answer_model


def generate_with_lora(prompt: str, max_new_tokens: int = 220) -> str:
    tokenizer, model = get_answer_model()

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    enc = {k: v.to(ANSWER_DEVICE) for k, v in enc.items()}

    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = gen[0][input_len:]
    pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return pred_text


def rewrite_query(state: RAGState) -> RAGState:
    print("\n[Agent] Rewriting query...")
    question = state["question"]

    prompt = (
        "Rewrite the user's question to improve retrieval from Steam reviews. "
        "Use keywords likely to appear in reviews.\n\n"
        f"QUESTION: {question}"
    )

    rewritten = llm.invoke(prompt).content.strip()

    state["rewritten_question"] = rewritten
    return state


def rebuild_docs_from_parquet(docs: list[Document]) -> list[Document]:
    rebuilt_docs = []

    for d in docs:
        d.metadata = d.metadata or {}
        chunk_id = (
            d.metadata.get("chunk_id")
            or d.metadata.get("id")
            or d.metadata.get("_id")
            or (d.page_content.strip() if isinstance(d.page_content, str) else None)
        )

        if not chunk_id:
            continue

        if chunk_id in chunks_df.index:
            text = chunks_df.loc[chunk_id, "text"]
            d.metadata["chunk_id"] = chunk_id
            rebuilt_docs.append(Document(page_content=str(text), metadata=d.metadata))

    return rebuilt_docs


def retrieve_docs(state: RAGState) -> RAGState:
    print("[Agent] Retrieving documents from Pinecone...")
    query = state["rewritten_question"]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(query)
    docs = rebuild_docs_from_parquet(docs)

    state["retrieved_docs"] = docs
    return state


def filter_relevant_docs(state: RAGState) -> RAGState:
    print("[Agent] Filtering relevant documents...")
    question = state["question"]
    docs = state["retrieved_docs"]

    if not docs:
        state["filtered_docs"] = []
        return state

    numbered_chunks = []
    for i, doc in enumerate(docs, 1):
        numbered_chunks.append(f"[DOC {i}]\n{doc.page_content}")

    prompt = (
        "You are judging which Steam review chunks are relevant to answering a question.\n"
        "Return only the document numbers that are relevant, separated by commas.\n"
        "If none are relevant, return NONE.\n\n"
        f"QUESTION: {question}\n\n"
        f"CHUNKS:\n\n" + "\n\n".join(numbered_chunks)
    )

    response = llm.invoke(prompt).content.strip()

    if response.upper() == "NONE":
        state["filtered_docs"] = []
        return state

    keep_indices = []
    for part in response.split(","):
        part = part.strip()
        if part.isdigit():
            keep_indices.append(int(part) - 1)

    filtered = [docs[i] for i in keep_indices if 0 <= i < len(docs)]
    state["filtered_docs"] = filtered
    return state


def decide_next_step(state: RAGState) -> RAGState:
    print("[Agent] Deciding next step...")
    docs = state["filtered_docs"]
    question = state["question"]

    prompt = (
        "You are deciding whether the system has enough relevant Steam review information "
        "to answer the user's question.\n"
        "If there is enough relevant information, return ANSWER.\n"
        "If there is not enough relevant information, return RETRIEVE.\n\n"
        f"QUESTION: {question}\n"
        f"NUMBER OF RELEVANT CHUNKS: {len(docs)}"
    )

    decision = llm.invoke(prompt).content.strip().upper()

    if "RETRIEVE" in decision:
        state["route_decision"] = "retrieve"
    else:
        state["route_decision"] = "answer"

    return state


def generate_answer_base(state: RAGState) -> RAGState:
    print("[Agent] Generating final answer with OpenAI model...")
    question = state["question"]
    docs = state["filtered_docs"]

    if not docs:
        state["answer"] = "I could not find relevant Steam review chunks to answer that question."
        return state

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

    prompt = (
        "You are answering questions using ONLY the provided Steam review chunks.\n"
        "Be specific and grounded in the evidence.\n"
        "Mention both praise and criticism when relevant.\n"
        "If the chunks do not contain enough information to answer, say so clearly.\n\n"
        f"QUESTION: {question}\n\n"
        f"STEAM REVIEW CHUNKS:\n{context}\n\n"
        "ANSWER:"
    )

    state["answer"] = llm.invoke(prompt).content.strip()
    return state


def generate_answer_lora(state: RAGState) -> RAGState:
    print("[Agent] Generating final answer with LoRA model...")
    question = state["question"]
    docs = state["filtered_docs"]

    if not docs:
        state["answer"] = "I could not find relevant Steam review chunks to answer that question."
        return state

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

    prompt = (
        "You are answering questions using ONLY the provided Steam review chunks.\n"
        "Be specific and grounded in the evidence.\n"
        "Mention both praise and criticism when relevant.\n"
        "If the chunks do not contain enough information to answer, say so clearly.\n\n"
        f"QUESTION: {question}\n\n"
        f"STEAM REVIEW CHUNKS:\n{context}\n\n"
        "ANSWER:"
    )

    state["answer"] = generate_with_lora(prompt)
    return state