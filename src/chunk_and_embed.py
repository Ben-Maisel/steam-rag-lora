import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv    # import libraries

from openai import OpenAI
import tiktoken


# -----------------------------
# Text helpers
# -----------------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)          # collapse mega newlines
    s = re.sub(r"[ \t]{2,}", " ", s)         # collapse whitespace
    return s.strip()

def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))


# -----------------------------
# Tokenization helpers
# -----------------------------

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))

def split_long_text(text: str, target_tokens: int, overlap_tokens: int, enc):
    tokens = enc.encode(text)
    if len(tokens) <= target_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)
    return chunks

def trim_to_tokens(text: str, max_tokens: int, enc) -> str:
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])


# -----------------------------
# Embedding
# -----------------------------

def embed_texts(client: OpenAI, texts, model: str, batch_size: int):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default="data/steam_reviews_out/ALL_STEAM_REVIEWS.csv",
        help="Input CSV of all scraped reviews (repo-root relative by default)."
    )
    parser.add_argument(
        "--outdir",
        default="data/embeddings",
        help="Output directory for parquet (repo-root relative by default)."
    )
    parser.add_argument(
        "--embed_model",
        default="text-embedding-3-small",
        help="OpenAI embedding model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Embedding batch size."
    )
    args = parser.parse_args()

    # Load env (OPENAI_API_KEY)
    load_dotenv()
    client = OpenAI()

    enc = tiktoken.get_encoding("cl100k_base")

    # Parameters that can be tuned
    SHORT_WORDS = 60
    PACK_MAX_TOKENS = 800
    LONG_REVIEW_TOKENS = 900
    SPLIT_TARGET_TOKENS = 450
    SPLIT_OVERLAP_TOKENS = 60
    MAX_EMBED_TOKENS = 7500

    # Paths
    infile = args.infile
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "steam_review_chunks_with_embeddings.parquet")

    print(f"Reading: {infile}")
    df = pd.read_csv(infile)

    # Filter & timestamp
    df = df[df["language"] == "english"].copy()
    df["created_at"] = pd.to_datetime(df["timestamp_created"], unit="s", utc=True)

    # Prep columns
    df["game_name"] = df["game_name"].astype(str)
    df["review_clean"] = df["review"].apply(clean_text)
    df["sentiment"] = np.where(df["voted_up"] == True, "positive", "negative")
    df["month_bucket"] = df["created_at"].dt.to_period("M").astype(str)  # e.g. "2025-01"
    df["review_words"] = df["review_clean"].apply(word_count)
    df["review_tokens"] = df["review_clean"].apply(lambda t: count_tokens(t, enc))

    chunks = []

    # 1) normal/long reviews first (>= SHORT_WORDS)
    normal_df = df[df["review_words"] >= SHORT_WORDS].copy()

    for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="Chunking normal/long reviews"):
        base_meta = {
            "appid": int(row["appid"]),
            "game_name": row["game_name"],
            "sentiment": row["sentiment"],
            "language": row["language"],
            "timestamp_created": int(row["timestamp_created"]),
            "created_at": row["created_at"].isoformat(),
            "votes_up": int(row.get("votes_up", 0)),
            "votes_funny": int(row.get("votes_funny", 0)),
            "comment_count": int(row.get("comment_count", 0)),
            "awards_received": int(row.get("awards_received", 0)),
            "playtime_at_review": int(row.get("playtime_at_review", 0)),
            "steam_purchase": bool(row.get("steam_purchase", False)),
            "received_for_free": bool(row.get("received_for_free", False)),
            "written_during_early_access": bool(row.get("written_during_early_access", False)),
            "recommendationid": str(row.get("recommendationid", "")),
            "chunk_type": "single_review",
            "month_bucket": row["month_bucket"],
        }

        text = row["review_clean"]

        if int(row["review_tokens"]) > LONG_REVIEW_TOKENS:
            parts = split_long_text(text, SPLIT_TARGET_TOKENS, SPLIT_OVERLAP_TOKENS, enc)
            for i, part in enumerate(parts):
                chunks.append({
                    **base_meta,
                    "chunk_id": f"{base_meta['appid']}_{base_meta['recommendationid']}_part{i+1}of{len(parts)}",
                    "chunk_index": i,
                    "chunk_count": len(parts),
                    "text": part,
                    "text_tokens": count_tokens(part, enc),
                    "review_ids": [base_meta["recommendationid"]],
                })
        else:
            chunks.append({
                **base_meta,
                "chunk_id": f"{base_meta['appid']}_{base_meta['recommendationid']}",
                "chunk_index": 0,
                "chunk_count": 1,
                "text": text,
                "text_tokens": int(row["review_tokens"]),
                "review_ids": [base_meta["recommendationid"]],
            })

    # 2) pack short reviews (< SHORT_WORDS) by game + sentiment + month
    short_df = df[df["review_words"] < SHORT_WORDS].copy()
    group_cols = ["appid", "game_name", "sentiment", "month_bucket"]

    for (appid, game_name, sentiment, month_bucket), g in tqdm(short_df.groupby(group_cols), desc="Packing short reviews"):
        g = g.sort_values(["votes_up", "comment_count"], ascending=False)

        current_texts = []
        current_ids = []
        current_tokens = 0
        pack_idx = 0

        for _, row in g.iterrows():
            rid = str(row.get("recommendationid", ""))
            txt = row["review_clean"]
            if not txt:
                continue

            review_block = f"[{rid}] {txt}"
            block_tokens = count_tokens(review_block, enc)

            if current_texts and (current_tokens + block_tokens) > PACK_MAX_TOKENS:
                pack_idx += 1
                chunks.append({
                    "chunk_id": f"{int(appid)}_{sentiment}_{month_bucket}_pack{pack_idx}",
                    "appid": int(appid),
                    "game_name": game_name,
                    "sentiment": sentiment,
                    "language": "english",
                    "timestamp_created": None,
                    "created_at": None,
                    "votes_up": None,
                    "votes_funny": None,
                    "comment_count": None,
                    "awards_received": None,
                    "playtime_at_review": None,
                    "steam_purchase": None,
                    "received_for_free": None,
                    "written_during_early_access": None,
                    "recommendationid": None,
                    "chunk_type": "packed_short_reviews",
                    "month_bucket": month_bucket,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "text": "\n".join(current_texts),
                    "text_tokens": current_tokens,
                    "review_ids": current_ids,
                })
                current_texts, current_ids, current_tokens = [], [], 0

            current_texts.append(review_block)
            current_ids.append(rid)
            current_tokens += block_tokens

        if current_texts:
            pack_idx += 1
            chunks.append({
                "chunk_id": f"{int(appid)}_{sentiment}_{month_bucket}_pack{pack_idx}",
                "appid": int(appid),
                "game_name": game_name,
                "sentiment": sentiment,
                "language": "english",
                "timestamp_created": None,
                "created_at": None,
                "votes_up": None,
                "votes_funny": None,
                "comment_count": None,
                "awards_received": None,
                "awards_received": None,
                "playtime_at_review": None,
                "steam_purchase": None,
                "received_for_free": None,
                "written_during_early_access": None,
                "recommendationid": None,
                "chunk_type": "packed_short_reviews",
                "month_bucket": month_bucket,
                "chunk_index": 0,
                "chunk_count": 1,
                "text": "\n".join(current_texts),
                "text_tokens": current_tokens,
                "review_ids": current_ids,
            })

    chunks_df = pd.DataFrame(chunks)
    print(f"Built {len(chunks_df)} chunks.")

    # Safety: cap chunk length before embedding
    chunks_df["text"] = chunks_df["text"].apply(lambda t: trim_to_tokens(t, MAX_EMBED_TOKENS, enc))
    chunks_df["text_tokens"] = chunks_df["text"].apply(lambda t: count_tokens(t, enc))

    max_tokens_seen = int(chunks_df["text_tokens"].max()) if len(chunks_df) else 0
    print(f"Max tokens in any chunk after trimming: {max_tokens_seen}")

    # Embed
    texts = chunks_df["text"].tolist()
    embeds = embed_texts(client, texts, model=args.embed_model, batch_size=args.batch_size)

    chunks_df["embedding"] = embeds
    chunks_df["embedding_dim"] = chunks_df["embedding"].apply(len)

    # Save
    chunks_df.to_parquet(out_path, index=False)
    dims = chunks_df["embedding_dim"].value_counts().to_dict()
    print(f"Saved: {out_path}")
    print(f"Embedding dims distribution: {dims}")


if __name__ == "__main__":
    main()
