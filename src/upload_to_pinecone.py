import os
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from pinecone import Pinecone

def build_metadata(row: pd.Series, vec_id: str) -> Dict[str, Any]:
    """
    Keep metadata SMALL. Do NOT store full text in Pinecone metadata.
    Store only what you might want to filter on + a reference id (chunk_id is the record id).
    """
    meta = {
        "chunk_id": vec_id,
        "appid": int(row["appid"]) if pd.notna(row.get("appid")) else None,
        "game_name": str(row["game_name"]) if pd.notna(row.get("game_name")) else None,
        "sentiment": str(row["sentiment"]) if pd.notna(row.get("sentiment")) else None,
        "month_bucket": str(row["month_bucket"]) if pd.notna(row.get("month_bucket")) else None,
        "chunk_type": str(row["chunk_type"]) if pd.notna(row.get("chunk_type")) else None,
    }

    # optional ranking/context fields (still small)
    for k in ["votes_up", "awards_received", "comment_count", "playtime_at_review"]:
        if k in row and pd.notna(row[k]):
            meta[k] = int(row[k])

    # remove Nones (Pinecone metadata should be clean JSON)
    return {k: v for k, v in meta.items() if v is not None}

def chunked(iterable: List[Tuple[str, List[float], Dict[str, Any]]], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]
        
def _strip_quotes(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1].strip()
    return s

        
def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default=os.getenv("EMBEDDINGS_PARQUET", "data/embeddings/steam_review_chunks_with_embeddings.parquet"),
        help="Path to parquet produced by chunking+embedding step."
    )
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "steam-reviews"))
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only upsert first N vectors (for testing).")
    ap.add_argument("--sample_frac", type=float, default=0.0, help="If >0, random sample fraction (e.g. 0.2).")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--reset_namespace", action="store_true", help="Delete all vectors in the namespace before uploading.")

    args = ap.parse_args()

    api_key = _strip_quotes(os.getenv("PINECONE_API_KEY"))
    host = _strip_quotes(os.getenv("PINECONE_INDEX_HOST"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "steam-reviews")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment/.env")

    # If host is missing, auto-resolve it from Pinecone
    if not host:
        pc = Pinecone(api_key=api_key)

        desc = pc.describe_index(index_name)
        if isinstance(desc, dict):
            host = desc.get("host") or desc.get("status", {}).get("host")
        else:
            host = getattr(desc, "host", None) or getattr(desc.status, "host", None)

    if not host:
        raise ValueError(
            "Missing PINECONE_INDEX_HOST and could not auto-resolve it via describe_index()."
        )

    # Load parquet
    df = pd.read_parquet(args.parquet)

    # Basic validation
    required_cols = {"chunk_id", "embedding", "game_name", "appid", "sentiment", "month_bucket", "chunk_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    # Optional subsampling (useful if you hit 2GB free-tier limit)
    if args.sample_frac and args.sample_frac > 0:
        df = df.sample(frac=args.sample_frac, random_state=args.random_seed).reset_index(drop=True)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)

    # Build pinecone client + index connection (host-targeting is recommended)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=host)

    if args.reset_namespace:
        print(f"[RESET] Deleting namespace '{args.namespace}' before upload")
        index.delete(delete_all=True, namespace=args.namespace)


    # Prepare upsert payload
    vectors = []
    for _, row in df.iterrows():
        vec_id = str(row["chunk_id"])
        values = row["embedding"]
        if not isinstance(values, list):
            # sometimes parquet can store as ndarray
            values = list(values)

        values = np.array(values, dtype=np.float32).tolist()
        
        vec_id = str(row["chunk_id"])
        metadata = build_metadata(row, vec_id)
        vectors.append((vec_id, values, metadata))

    # (Optional) print embedding dimension
    dim = len(vectors[0][1]) if vectors else 0
    print(f"Ready to upsert {len(vectors)} vectors. Embedding dimension = {dim}")
    print(f"Namespace = {args.namespace}")

    # Upsert in batches
    for batch in tqdm(list(chunked(vectors, args.batch_size)), desc="Upserting to Pinecone"):
        index.upsert(vectors=batch, namespace=args.namespace)

    print("Done upserting.")


if __name__ == "__main__":
    main()