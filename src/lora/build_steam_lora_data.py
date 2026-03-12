import os
import json
import uuid
import random
import argparse
import re
import math
from pathlib import Path

import pandas as pd


QUESTION_BANK = {
    "Black Myth: Wukong": [
        "What do players think of the graphics in Black Myth: Wukong?",
        "Is Black Myth: Wukong visually impressive?",
        "How do players describe the art direction and visuals in Black Myth: Wukong?",
        "Do players praise the presentation in Black Myth: Wukong?",
        "How good do the visuals look in Black Myth: Wukong according to reviews?",
    ],
    "TEKKEN 8": [
        "Is TEKKEN 8 balanced?",
        "What do players think about the balance in TEKKEN 8?",
        "Do players think TEKKEN 8 is fair and competitive?",
        "How do players describe character balance in TEKKEN 8?",
        "Do reviews suggest TEKKEN 8 feels fair online?",
    ],
    "Blue Prince": [
        "What do people think of Blue Prince?",
        "What is the gameplay loop of Blue Prince like?",
        "How do players describe the overall experience of Blue Prince?",
        "What do reviews say about the structure of Blue Prince?",
        "Do players find Blue Prince engaging over time?",
    ],
    "Frostpunk 2": [
        "What is the gameplay loop of Frostpunk 2 like?",
        "How do players describe the gameplay in Frostpunk 2?",
        "What do players think about managing systems in Frostpunk 2?",
        "Do players enjoy the strategy systems in Frostpunk 2?",
        "How does Frostpunk 2 feel to play according to reviews?",
    ],
    "Kingdom Come: Deliverance 2": [
        "Is Kingdom Come: Deliverance 2 worth it?",
        "Do players think Kingdom Come: Deliverance 2 is worth buying?",
        "What do players like and dislike about Kingdom Come: Deliverance 2?",
        "Would players recommend Kingdom Come: Deliverance 2?",
        "How do reviews describe the value of Kingdom Come: Deliverance 2?",
    ],
    "Monster Hunter Wilds": [
        "What do players think about the difficulty of Monster Hunter Wilds?",
        "Is Monster Hunter Wilds difficult according to players?",
        "How do players describe the challenge level in Monster Hunter Wilds?",
        "Do players think Monster Hunter Wilds is too hard or too easy?",
        "How demanding do reviews say Monster Hunter Wilds is?",
    ],
    "Clair Obscur: Expedition 33": [
        "Is the music in Clair Obscur: Expedition 33 good?",
        "What do players think about the soundtrack in Clair Obscur: Expedition 33?",
        "How do players describe the music in Clair Obscur: Expedition 33?",
        "Do reviews praise the audio and soundtrack in Clair Obscur: Expedition 33?",
        "How strong is the music in Clair Obscur: Expedition 33 according to players?",
    ],
}

TOPIC_KEYWORDS = {
    "graphics": {"graphics", "visuals", "beautiful", "gorgeous", "art", "artstyle", "visual", "look", "looks", "presentation"},
    "music": {"music", "soundtrack", "audio", "score", "songs", "sound"},
    "balance": {"balance", "balanced", "fair", "unfair", "competitive", "ranked", "characters", "matchup"},
    "gameplay loop": {"loop", "gameplay", "mechanics", "systems", "progression", "core", "play", "replayability", "strategy"},
    "worth it": {"worth", "buy", "price", "value", "recommend", "recommended"},
    "difficulty": {"difficulty", "difficult", "hard", "easy", "challenge", "challenging"},
    "overall": {"good", "bad", "fun", "boring", "recommend", "worth", "enjoy", "experience", "engaging"},
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "for", "to", "do", "does",
    "what", "how", "about", "according", "players", "player", "think", "describe",
    "and", "it", "this", "that", "like", "reviews", "review", "say", "suggest"
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def infer_topic(question: str) -> str:
    q = question.lower()
    if "graphic" in q or "visual" in q or "art" in q or "presentation" in q:
        return "graphics"
    if "music" in q or "soundtrack" in q or "audio" in q or "score" in q:
        return "music"
    if "balance" in q or "balanced" in q or "fair" in q or "competitive" in q:
        return "balance"
    if "loop" in q or "gameplay" in q or "mechanic" in q or "progression" in q or "system" in q:
        return "gameplay loop"
    if "worth" in q or "buy" in q or "price" in q or "value" in q or "recommend" in q:
        return "worth it"
    if "difficulty" in q or "difficult" in q or "hard" in q or "easy" in q or "challenge" in q:
        return "difficulty"
    return "overall"


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def best_sentences(text: str, topic: str, max_sentences: int = 2) -> list[str]:
    topic_words = TOPIC_KEYWORDS.get(topic, set())
    sentences = sentence_split(text)
    scored = []

    for s in sentences:
        toks = set(tokenize(s))
        score = len(toks & topic_words)
        if len(s) < 25:
            continue
        scored.append((score, len(s), s))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    chosen = [s for _, _, s in scored[:max_sentences]]

    if not chosen:
        for s in sentences[:max_sentences]:
            if len(s) >= 25:
                chosen.append(s)

    return chosen[:max_sentences]


def score_chunk(text: str, question: str, topic: str) -> int:
    text_tokens = set(tokenize(text))
    q_tokens = {t for t in tokenize(question) if t not in STOPWORDS}
    topic_words = TOPIC_KEYWORDS.get(topic, set())

    score = 0
    score += 3 * len(text_tokens & topic_words)
    score += 1 * len(text_tokens & q_tokens)

    if "recommend" in text_tokens:
        score += 1
    if "worth" in text_tokens and topic == "worth it":
        score += 2
    if "difficult" in text_tokens and topic == "difficulty":
        score += 2
    if "balanced" in text_tokens and topic == "balance":
        score += 2

    return score


def build_prompt(game: str, question: str, docs: list[dict]) -> str:
    context_blocks = []
    for i, d in enumerate(docs, 1):
        sentiment = d.get("sentiment", "unknown")
        chunk_id = d.get("chunk_id", f"chunk_{i}")
        text = d.get("text", "").strip()
        context_blocks.append(
            f"[CHUNK {i}] chunk_id={chunk_id} game={game} sentiment={sentiment}\n{text}"
        )

    context = "\n\n".join(context_blocks)

    return (
        "Answer the question using only the provided Steam review chunks.\n"
        "Be specific, grounded in the reviews, and mention both praise and criticism when relevant.\n"
        "If the chunks are mixed, reflect that clearly.\n\n"
        f"QUESTION: {question}\n\n"
        f"STEAM REVIEW CHUNKS:\n{context}\n\n"
        "ANSWER:"
    )


def build_response(game: str, question: str, docs: list[dict], topic: str) -> str:
    pos = sum(1 for d in docs if d.get("sentiment") == "positive")
    neg = sum(1 for d in docs if d.get("sentiment") == "negative")

    if pos > neg:
        opening = "Overall, the reviews lean positive."
    elif neg > pos:
        opening = "Overall, the reviews lean negative."
    else:
        opening = "Overall, the reviews are mixed."

    topic_sentence = {
        "graphics": f"For {game}, players frequently talk about the visuals and presentation.",
        "music": f"For {game}, players frequently comment on the music and audio experience.",
        "balance": f"For {game}, players frequently discuss game balance and fairness.",
        "gameplay loop": f"For {game}, players focus heavily on the core gameplay loop and progression.",
        "worth it": f"For {game}, players often frame their opinion in terms of value and whether the game is worth buying.",
        "difficulty": f"For {game}, players frequently describe the challenge level and how demanding the game feels.",
        "overall": f"For {game}, players comment on the overall experience.",
    }[topic]

    evidence_lines = []
    for d in docs[:4]:
        for s in best_sentences(d.get("text", ""), topic, max_sentences=1):
            evidence_lines.append(s)
            break

    evidence_lines = evidence_lines[:4]

    if evidence_lines:
        evidence_paragraph = "Review excerpts mention that " + " ".join(evidence_lines)
    else:
        evidence_paragraph = (
            "The retrieved review chunks provide enough information to form a grounded summary, "
            "but the evidence is more general than highly specific."
        )

    if pos > 0 and neg > 0:
        closing = (
            "Taken together, the response should reflect that some players are enthusiastic while others point out drawbacks."
        )
    elif pos > neg:
        closing = "Taken together, most of the retrieved reviews suggest players are satisfied overall."
    else:
        closing = "Taken together, the retrieved reviews suggest notable complaints or limitations are present."

    return f"{opening} {topic_sentence} {evidence_paragraph} {closing}"


def choose_docs_for_variant(scored_docs: list[dict], variant_idx: int, docs_per_example: int = 6) -> list[dict]:
    """
    Create slightly different top-doc selections so repeated question templates
    still produce different training examples.
    """
    if not scored_docs:
        return []

    start = min(variant_idx * 2, max(0, len(scored_docs) - docs_per_example))
    window = scored_docs[start:start + docs_per_example]

    if len(window) < docs_per_example:
        window = scored_docs[:docs_per_example]

    return window


def sample_examples_for_game(
    game_df: pd.DataFrame,
    game: str,
    examples_per_game: int,
) -> list[dict]:
    rows = []
    questions = QUESTION_BANK.get(game, [])
    if not questions:
        return rows

    for example_idx in range(examples_per_game):
        question = questions[example_idx % len(questions)]
        topic = infer_topic(question)

        scored_docs = []
        for _, row in game_df.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            score = score_chunk(text, question, topic)
            scored_docs.append(
                {
                    "score": score,
                    "chunk_id": row["chunk_id"],
                    "sentiment": row.get("sentiment", "unknown"),
                    "text": text,
                }
            )

        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        top_docs = choose_docs_for_variant(
            scored_docs=scored_docs,
            variant_idx=example_idx,
            docs_per_example=6,
        )

        if not top_docs:
            continue

        prompt = build_prompt(game, question, top_docs)
        response = build_response(game, question, top_docs, topic)

        rows.append(
            {
                "id": f"steam_{uuid.uuid4().hex[:12]}",
                "category": "steam_rag_qa",
                "prompt": prompt,
                "response": response,
            }
        )

    return rows


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default="data/embeddings/steam_review_chunks_with_embeddings.parquet",
        help="Path to chunk parquet file.",
    )
    parser.add_argument(
        "--train_out",
        default="data/lora/steam_train.jsonl",
        help="Path to output train JSONL.",
    )
    parser.add_argument(
        "--eval_out",
        default="data/lora/steam_eval.jsonl",
        help="Path to output eval JSONL.",
    )
    parser.add_argument(
        "--examples_per_game",
        type=int,
        default=9,
        help="Initial target number of QA examples to generate per game.",
    )
    parser.add_argument(
        "--eval_frac",
        type=float,
        default=0.15,
        help="Fraction of examples to place into eval.",
    )
    parser.add_argument(
        "--min_train_examples",
        type=int,
        default=50,
        help="Minimum number of training examples required after the split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(f"Could not find parquet file at: {infile}")

    df = pd.read_parquet(infile)

    required_cols = {"chunk_id", "game_name", "sentiment", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing required columns: {missing}")

    games = sorted(df["game_name"].dropna().unique())
    if not games:
        raise ValueError("No games found in the parquet file.")

    examples_per_game = args.examples_per_game

    while True:
        examples = []

        for game in games:
            game_df = df[df["game_name"] == game].copy()
            game_examples = sample_examples_for_game(
                game_df=game_df,
                game=game,
                examples_per_game=examples_per_game,
            )
            examples.extend(game_examples)

        random.shuffle(examples)

        if not examples:
            raise ValueError("No examples were generated.")

        eval_size = max(5, int(round(len(examples) * args.eval_frac)))
        train_size = len(examples) - eval_size

        if train_size >= args.min_train_examples:
            break

        examples_per_game += 1

        if examples_per_game > 50:
            raise ValueError(
                "Could not reach the requested minimum training size. "
                "Check the source data and question bank."
            )

    eval_records = examples[:eval_size]
    train_records = examples[eval_size:]

    os.makedirs(Path(args.train_out).parent, exist_ok=True)
    write_jsonl(args.train_out, train_records)
    write_jsonl(args.eval_out, eval_records)

    print(f"Games found: {len(games)}")
    print(f"Examples per game used: {examples_per_game}")
    print(f"Wrote {len(train_records)} train examples to {args.train_out}")
    print(f"Wrote {len(eval_records)} eval examples to {args.eval_out}")
    print(f"Total examples: {len(examples)}")


if __name__ == "__main__":
    main()