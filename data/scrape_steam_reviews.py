import csv
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple    # import libraries
import requests


# The code below is just webscrapping code to gather the most recent reviews of video games from the steam store.
# The reviews and the meta data is stored in data/steam_reviews_out/ once the scrapping is complete.
# A lot of data is gathered so please be patient. If the data exists already, Docker will skip this step.

GAMES = [
    "Black Myth: Wukong",
    "Like a Dragon: Infinite Wealth",   # games to scrape
    "Frostpunk 2",
    "TEKKEN 8",
    "Warhammer 40,000: Space Marine 2",
    "Balatro",
    "Monster Hunter Wilds",
    "Assassin's Creed Shadows",
    "Split Fiction",
    "Battlefield 6",
    "Blue Prince",
    "Kingdom Come: Deliverance 2",
    "Hollow Knight: Silksong",
    "Clair Obscur: Expedition 33",
]

USER_AGENT = "Mozilla/5.0 (compatible; class-rag-scraper/1.0; +https://example.edu)"

@dataclass
class ReviewRow:
    appid: int
    game_name: str
    recommendationid: str
    voted_up: bool
    language: str
    timestamp_created: int
    timestamp_updated: int
    review: str

    # usefulness
    votes_up: int
    votes_funny: int
    weighted_vote_score: str
    comment_count: int

    # purchase & context
    steam_purchase: bool
    received_for_free: bool
    written_during_early_access: bool

    # playtime
    playtime_forever: int
    playtime_at_review: int
    playtime_last_two_weeks: int

    # awards (if present)
    awards_received: int

    # bookkeeping
    scrape_timestamp: int
    

def http_get_json(session: requests.Session, url: str, params: Dict, max_retries: int = 5) -> Dict:
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                # rate limited
                sleep_s = 2 ** attempt + random.random()
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries:
                raise
            sleep_s = 1.5 ** attempt + random.random()
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")

def resolve_appid(session: requests.Session, game_name: str, cc: str = "US", lang: str = "english") -> Optional[int]:
    """
    Uses the public Steam store search API to find the appid for a game name.
    """
    url = "https://store.steampowered.com/api/storesearch/"
    params = {
        "term": game_name,
        "cc": cc,
        "l": lang,
    }
    data = http_get_json(session, url, params)

    items = data.get("items", [])
    if not items:
        return None

    return int(items[0]["id"])

def fetch_reviews_page(
    session: requests.Session,
    appid: int,
    cursor: str,
    language: str = "all",
    filter_mode: str = "recent",
    num_per_page: int = 100,
) -> Dict:
    """
    Steam public reviews endpoint. Cursor-based pagination.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        "json": 1,
        "filter": filter_mode,           # "recent", "all", "updated", "funny", "helpful"
        "language": language,            # "all" or specific like "english"
        "purchase_type": "all",
        "num_per_page": num_per_page,    # max 100
        "cursor": cursor,
    }
    return http_get_json(session, url, params)

def parse_review(appid: int, game_name: str, r: Dict, scrape_ts: int) -> ReviewRow:
    author = r.get("author", {}) or {}

    # Awards field can vary; handle gracefully.
    awards = r.get("awards_received")
    if isinstance(awards, int):
        awards_received = awards
    elif isinstance(awards, dict):
        # sometimes it's a dict by reaction type; sum it
        awards_received = int(sum(awards.values()))
    else:
        awards_received = 0

    return ReviewRow(
        appid=appid,
        game_name=game_name,
        recommendationid=str(r.get("recommendationid", "")),
        voted_up=bool(r.get("voted_up", False)),
        language=str(r.get("language", "")),
        timestamp_created=int(r.get("timestamp_created", 0)),
        timestamp_updated=int(r.get("timestamp_updated", 0)),
        review=str(r.get("review", "")),

        votes_up=int(r.get("votes_up", 0)),
        votes_funny=int(r.get("votes_funny", 0)),
        weighted_vote_score=str(r.get("weighted_vote_score", "")),
        comment_count=int(r.get("comment_count", 0)),

        steam_purchase=bool(r.get("steam_purchase", False)),
        received_for_free=bool(r.get("received_for_free", False)),
        written_during_early_access=bool(r.get("written_during_early_access", False)),

        playtime_forever=int(author.get("playtime_forever", 0)),
        playtime_at_review=int(author.get("playtime_at_review", 0)),
        playtime_last_two_weeks=int(author.get("playtime_last_two_weeks", 0)),

        awards_received=awards_received,
        scrape_timestamp=scrape_ts,
    )

def write_csv(path: str, rows: List[ReviewRow]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row.__dict__)
            

def scrape_game(
    session: requests.Session,
    game_name: str,
    appid: int,
    max_reviews: int,
    filter_mode: str,
    sleep_min: float,
    sleep_max: float,
) -> List[ReviewRow]:
    rows: List[ReviewRow] = []
    cursor = "*"  # initial cursor
    scrape_ts = int(time.time())

    while len(rows) < max_reviews:
        page = fetch_reviews_page(session, appid, cursor=cursor, filter_mode=filter_mode)
        if page.get("success") != 1:
            break

        reviews = page.get("reviews", []) or []
        if not reviews:
            break

        for r in reviews:
            rows.append(parse_review(appid, game_name, r, scrape_ts))
            if len(rows) >= max_reviews:
                break

        cursor = page.get("cursor", cursor)
        time.sleep(random.uniform(sleep_min, sleep_max))

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/steam_reviews_out", help="Output folder name (relative to cwd).")
    ap.add_argument("--max_reviews_per_game", type=int, default=2000, help="How many reviews to scrape per game.")
    ap.add_argument("--filter", default="recent", choices=["recent", "all", "updated", "helpful", "funny"],
                    help="Steam filter mode for reviews.")
    ap.add_argument("--sleep_min", type=float, default=0.4, help="Min sleep between pages (seconds).")
    ap.add_argument("--sleep_max", type=float, default=1.2, help="Max sleep between pages (seconds).")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # 1) Resolve appids
    name_to_appid: Dict[str, int] = {}
    unresolved: List[str] = []

    for name in GAMES:
        appid = resolve_appid(session, name)
        if appid is None:
            unresolved.append(name)
        else:
            name_to_appid[name] = appid

    # Save mapping so you can cite it / reuse it
    mapping_path = os.path.join(args.outdir, "game_appids.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(name_to_appid, f, indent=2)

    if unresolved:
        print("\nCould not resolve these game names (try editing names to match Steam store titles):")
        for n in unresolved:
            print(" -", n)
        print("\nContinuing with resolved games only.\n")

    # 2) Scrape reviews per resolved game
    all_rows: List[ReviewRow] = []

    for game_name, appid in name_to_appid.items():
        print(f"Scraping: {game_name} (appid={appid}) ...")
        rows = scrape_game(
            session=session,
            game_name=game_name,
            appid=appid,
            max_reviews=args.max_reviews_per_game,
            filter_mode=args.filter,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
        )
        print(f"  -> got {len(rows)} reviews")

        safe_name = "".join(c for c in game_name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        out_csv = os.path.join(args.outdir, f"{safe_name}_{appid}.csv")
        write_csv(out_csv, rows)
        all_rows.extend(rows)

    # 3) Combined CSV
    if all_rows:
        combined_path = os.path.join(args.outdir, "ALL_STEAM_REVIEWS.csv")
        write_csv(combined_path, all_rows)
        print(f"\nWrote combined file: {combined_path}")
    print(f"Wrote appid mapping: {mapping_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()