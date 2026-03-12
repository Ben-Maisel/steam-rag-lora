#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /app/data/steam_reviews_out/ALL_STEAM_REVIEWS.csv ]; then
  echo "== Step 1: Scrape Steam reviews =="
  python /app/data/scrape_steam_reviews.py \
    --outdir /app/data/steam_reviews_out \
    --max_reviews_per_game 2000 \
    --filter recent
else
  echo "== Step 1: Skipping scrape (CSV already exists) =="
fi

if [ ! -f /app/data/embeddings/steam_review_chunks_with_embeddings.parquet ]; then
  echo "== Step 2: Chunk and embed reviews =="
  python /app/src/chunk_and_embed.py \
    --infile /app/data/steam_reviews_out/ALL_STEAM_REVIEWS.csv \
    --outdir /app/data/embeddings \
    --embed_model text-embedding-3-small \
    --batch_size 200
else
  echo "== Step 2: Skipping chunk+embed (parquet already exists) =="
fi

echo "== Step 3: Setup Pinecone =="
python /app/src/setup_pinecone_env.py

export PINECONE_INDEX_HOST="$(python /app/src/setup_pinecone_env.py --print_host 2>/dev/null | tail -n 1)"
echo "Exported PINECONE_INDEX_HOST=$PINECONE_INDEX_HOST"

echo "== Step 4: Upload vectors to Pinecone =="
python /app/src/upload_to_pinecone.py \
  --parquet /app/data/embeddings/steam_review_chunks_with_embeddings.parquet \
  --namespace steam-reviews \
  --batch_size 200

echo "== Step 5: Run assignment test questions =="
python -m src.agents.agentic_rag --mode test

echo "== Step 6: Start interactive mode =="
python -m src.agents.agentic_rag --mode interactive