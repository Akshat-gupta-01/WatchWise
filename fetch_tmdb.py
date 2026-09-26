"""
fetch_tmdb.py
=============
Optional: rebuild ``data/titles.json`` from the live TMDB API.

The local catalog is committed so the app runs with no API key at all. Run this
only when you want fresher or larger data:

    export TMDB_API_KEY=your_key
    python fetch_tmdb.py --pages 5          # ~1000 titles -> data/titles.json

Requires ``requests`` (already a dependency). Writing to data/titles.json
replaces the curated catalog, including the ``vibes`` tags the mood search
depends on - the generator assigns those heuristically, so review the output
before using it in anger.

TMDB attribution: this product uses the TMDB API but is not endorsed or
certified by TMDB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("This script needs requests:  pip install requests")

API_KEY = os.getenv("TMDB_API_KEY", "").strip()
BASE = "https://api.themoviedb.org/3"
OUT = Path(__file__).resolve().parent / "data" / "titles.json"

GENRE_IDS = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action",
    10765: "Science Fiction", 10768: "War", 37: "Western",
}

# TMDB language code -> the label this project uses
LANGUAGES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ko": "Korean", "ja": "Japanese",
    "fr": "French", "de": "German", "es": "Spanish",
}

# Coarse vibes inferred from genre + keywords + popularity, so the mood tool has
# something to score against. Deliberately simple.
VIBE_RULES: list[tuple[set[str], list[str]]] = [
    ({"Comedy"}, ["funny", "light", "feel-good"]),
    ({"Romance"}, ["romantic", "warm"]),
    ({"Horror"}, ["scary", "creepy", "tense"]),
    ({"Thriller", "Mystery", "Crime"}, ["tense", "suspenseful", "twisty"]),
    ({"Science Fiction"}, ["mind-bending", "cerebral", "futuristic"]),
    ({"Action", "Adventure"}, ["exciting", "action-packed", "adventurous"]),
    ({"Documentary"}, ["reflective", "realistic"]),
    ({"Animation", "Family"}, ["cozy", "family", "whimsical"]),
    ({"Drama"}, ["emotional", "character-study"]),
    ({"War", "History"}, ["epic", "gritty", "historical"]),
]


def infer_vibes(genres: list[str], overview: str, rating: float) -> list[str]:
    vibes: list[str] = []
    genre_set = set(genres)
    for targets, tags in VIBE_RULES:
        if genre_set & targets:
            vibes.extend(tags)
    blob = overview.lower()
    hints = {
        "feel-good": ("feel good", "heartwarming", "uplifting", "friendship"),
        "bittersweet": ("bittersweet", "loss", "grief", "farewell"),
        "cozy": ("cozy", "quiet", "small town", "village"),
        "stylish": ("stylish", "neon", "cyberpunk"),
        "gritty": ("gritty", "crime", "gangster", "corrupt"),
        "mind-bending": ("mystery", "secret", "identity", "twist"),
    }
    for tag, needles in hints.items():
        if any(n in blob for n in needles):
            vibes.append(tag)
    if rating >= 8.0 and "critically acclaimed" not in vibes:
        vibes.append("critically acclaimed")
    return sorted(set(vibes))[:6]


def fetch(path: str, params: dict[str, Any]) -> dict[str, Any]:
    params = {**params, "api_key": API_KEY, "language": "en-US"}
    for attempt in range(4):
        resp = requests.get(f"{BASE}{path}", params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (2**attempt))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"TMDB kept failing for {path}")


def collect(pages: int, kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        path = "/discover/movie" if kind == "movie" else "/discover/tv"
        data = fetch(
            path,
            {
                "sort_by": "vote_average.desc",
                "vote_count.gte": 800 if kind == "movie" else 150,
                "page": page,
                "with_original_language": "|".join(LANGUAGES),
            },
        )
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            genres = [GENRE_IDS[g] for g in item.get("genre_ids", []) if g in GENRE_IDS]
            overview = (item.get("overview") or "").strip()
            rating = float(item.get("vote_average") or 0)
            if not overview:
                continue
            out.append(
                {
                    "id": f"tmdb-{kind}-{item['id']}",
                    "title": (item.get("title") or item.get("name") or "Untitled").strip(),
                    "year": int((item.get("release_date") or item.get("first_air_date") or "0000")[:4] or 0),
                    "type": kind,
                    "genres": sorted(set(genres)),
                    "language": LANGUAGES.get(item.get("original_language"), "English"),
                    "runtime_minutes": int(item.get("runtime") or (40 if kind == "series" else 110)),
                    "seasons": len(item.get("seasons") or []) or (1 if kind == "series" else None),
                    "overview": overview,
                    "director": "",
                    "cast": [c.get("name", "") for c in (item.get("credits") or {}).get("cast", [])][:3],
                    "rating": round(rating, 1),
                    "maturity": "",
                    "vibes": infer_vibes(sorted(set(genres)), overview, rating),
                }
            )
        print(f"  {kind}: page {page} -> {len(out)} titles", file=sys.stderr)
        time.sleep(0.25)  # be polite to the free tier
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild CineAgent's catalog from TMDB.")
    parser.add_argument("--pages", type=int, default=3, help="pages per media type (20 titles each)")
    parser.add_argument("--movies", action="store_true", help="movies only")
    parser.add_argument("--series", action="store_true", help="series only")
    args = parser.parse_args()

    if not API_KEY:
        print("TMDB_API_KEY is not set.\n\n"
              "The app already ships with data/titles.json and needs no key.\n"
              "Get a free key at https://www.themoviedb.org/settings/api", file=sys.stderr)
        return 1

    titles: list[dict[str, Any]] = []
    if not args.series:
        titles += collect(args.pages, "movie")
    if not args.movies:
        titles += collect(args.pages, "series")

    titles = [t for t in titles if t["year"] > 1960]
    titles.sort(key=lambda t: -(t["rating"] or 0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "metadata": {
                    "source": "tmdb",
                    "note": "Generated by fetch_tmdb.py. Vibes are heuristically inferred.",
                    "count": len(titles),
                },
                "titles": titles,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {len(titles)} titles to {OUT}", file=sys.stderr)
    print("Review the vibe tags before trusting mood search.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
