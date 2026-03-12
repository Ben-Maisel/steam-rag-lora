import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

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



def generate_answer(state: RAGState) -> RAGState:
    print("[Agent] Generating final answer...")
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
        "If the chunks do not contain enough information to answer, say so clearly.\n\n"
        f"QUESTION: {question}\n\n"
        f"STEAM REVIEW CHUNKS:\n{context}\n\n"
        "ANSWER:"
    )

    state["answer"] = llm.invoke(prompt).content.strip()
    return state