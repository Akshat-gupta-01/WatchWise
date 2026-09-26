"""
catalog.py
==========
The search engine behind CineAgent's tools.

Responsibilities:
  * Load the dataset (local ``data/titles.json``, or live TMDB if an API key is set)
  * Normalise every record into one consistent schema
  * Provide fuzzy, forgiving lookup helpers so the LLM's tool arguments
    ("bollywood", "hollywood", "short", "hi") never fall through the floor

Nothing in here talks to the LLM. tools.py wraps this module; llm.py decides
which tool to call. Keeping that boundary sharp is what makes the whole app
testable without a single API key.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parent / "data"
LOCAL_DATASET = DATA_DIR / "titles.json"

# --------------------------------------------------------------------------- #
# Normalisation maps
# --------------------------------------------------------------------------- #

# Synonyms so the LLM can say "sci-fi", "scifi", "science fiction" and all hit
# the same bucket.
GENRE_ALIASES: dict[str, str] = {
    "sci fi": "Science Fiction",
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "science fiction": "Science Fiction",
    "sf": "Science Fiction",
    "comedy": "Comedy",
    "funny": "Comedy",
    "comedies": "Comedy",
    "rom com": "Romance",
    "romance": "Romance",
    "romantic": "Romance",
    "thriller": "Thriller",
    "mystery": "Mystery",
    "crime": "Crime",
    "drama": "Drama",
    "action": "Action",
    "adventure": "Adventure",
    "horror": "Horror",
    "scary": "Horror",
    "fantasy": "Fantasy",
    "family": "Family",
    "animation": "Animation",
    "animated": "Animation",
    "anime": "Animation",
    "documentary": "Documentary",
    "doc": "Documentary",
    "docs": "Documentary",
    "sport": "Sport",
    "sports": "Sport",
    "war": "War",
    "historical": "Historical",
    "history": "History",
    "musical": "Musical",
    "music": "Music",
    "western": "Western",
    "classic": "Classic",
    "period": "Historical",
    "biopic": "Biography",
    "biography": "Biography",
    "true story": "Biography",
    "noir": "Crime",
    "psychological": "Thriller",
    "supernatural": "Fantasy",
    "biopic / history": "Biography",
}

# Language aliases: the LLM (and users) say "Hindi", "hindi film", "hi",
# "Bollywood", "Hollywood", "K-drama" ... all must resolve.
LANGUAGE_ALIASES: dict[str, str] = {
    "hi": "Hindi",
    "hindi": "Hindi",
    "hindi film": "Hindi",
    "hindi movie": "Hindi",
    "bollywood": "Hindi",
    "indian": "Hindi",
    "india": "Hindi",
    "en": "English",
    "eng": "English",
    "english": "English",
    "hollywood": "English",
    "american": "English",
    "western": "English",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "ja": "Japanese",
    "japanese": "Japanese",
    "anime": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "k-drama": "Korean",
    "kdramas": "Korean",
    "kn": "Kannada",
    "kannada": "Kannada",
    "te": "Telugu",
    "telugu": "Telugu",
    "ta": "Tamil",
    "tamil": "Tamil",
    "ml": "Malayalam",
    "malayalam": "Malayalam",
    "pa": "Punjabi",
    "punjabi": "Punjabi",
    "pa/pj": "Punjabi",
    "multi": "Multi",
    "multilingual": "Multi",
}

# Mood -> (genres, vibes, keywords). The LLM can pass any of these; we score a
# title by how strongly it matches. This is the knowledge that makes
# "something light" mean something concrete.
MOOD_MAP: dict[str, dict[str, Any]] = {
    "light": {
        "label": "light & easy",
        "genres": ["Comedy", "Romance"],
        "vibes": ["light", "breezy", "feel-good", "funny", "cozy", "fun", "mindless", "whimsical"],
        "keywords": ["comedy", "romance", "musical", "animation", "family", "quirky"],
        "avoid": ["gory", "gritty", "brutal", "scary", "noir", "bleak", "uncomfortable"],
    },
    "funny": {
        "label": "funny",
        "genres": ["Comedy"],
        "vibes": ["funny", "fun", "witty", "deadpan", "chatty", "quirky", "dark-funny"],
        "keywords": ["comedy", "satirical", "campy", "noisy"],
    },
    "feel-good": {
        "label": "feel-good",
        "genres": ["Comedy", "Family", "Adventure"],
        "vibes": ["feel-good", "uplifting", "heartfelt", "hopeful", "warm", "touching", "inspiring"],
        "keywords": ["inspirational", "underdog"],
    },
    "cozy": {
        "label": "cozy & comforting",
        "genres": ["Family", "Animation", "Romance"],
        "vibes": ["cozy", "comforting", "gentle", "warm", "nostalgic", "charming", "quiet"],
        "keywords": ["documentary", "nature"],
    },
    "romantic": {
        "label": "romantic",
        "genres": ["Romance", "Drama"],
        "vibes": ["romantic", "feel-good", "bittersweet", "quiet", "nostalgic"],
        "keywords": ["rom com", "coming-of-age"],
    },
    "emotional": {
        "label": "emotional & heartfelt",
        "genres": ["Drama"],
        "vibes": ["emotional", "heartfelt", "bittersweet", "touching", "weepy", "bittersweet"],
        "keywords": ["family", "friendship", "biopic"],
    },
    "inspiring": {
        "label": "inspiring",
        "genres": ["Drama", "Sport", "Biography"],
        "vibes": ["uplifting", "inspiring", "hopeful", "underdog", "touching", "stylized"],
        "keywords": ["sport", "war", "inspirational"],
    },
    "thrilling": {
        "label": "thrilling",
        "genres": ["Thriller", "Crime", "Action"],
        "vibes": ["tense", "suspenseful", "thrilling", "fast-paced", "intense", "gripping"],
        "keywords": ["chase", "spy", "heist"],
    },
    "twisty": {
        "label": "twisty & mind-bending",
        "genres": ["Mystery", "Thriller", "Science Fiction"],
        "vibes": ["twisty", "mind-bending", "cerebral", "clever", "suspenseful", "mysterious"],
        "keywords": ["mystery", "neo-noir"],
    },
    "dark": {
        "label": "dark & gritty",
        "genres": ["Crime", "Thriller", "War", "Drama"],
        "vibes": ["dark", "gritty", "noir", "bleak", "violent", "uncomfortable", "bleak"],
        "keywords": ["neo-noir", "gothic"],
    },
    "scary": {
        "label": "scary",
        "genres": ["Horror"],
        "vibes": ["scary", "creepy", "spooky", "unsettling", "folk-horror", "gothic", "gory"],
        "keywords": ["zombie", "gothic", "possession"],
        "avoid": ["family"],
    },
    "intense": {
        "label": "intense & hard-hitting",
        "genres": ["Action", "War", "Crime"],
        "vibes": ["intense", "brutal", "gruelling", "gritty", "gruelling", "gory", "violent"],
        "keywords": ["war", "revenge", "survival"],
    },
    "action": {
        "label": "action & adrenaline",
        "genres": ["Action", "Adventure"],
        "vibes": ["action-packed", "exciting", "slick", "stylish", "spectacle", "thrilling"],
        "keywords": ["heist", "spy", "martial"],
    },
    "adventurous": {
        "label": "adventurous",
        "genres": ["Adventure", "Action", "Science Fiction"],
        "vibes": ["adventurous", "epic", "sweeping", "spectacle", "exploring"],
        "keywords": ["journey", "quest"],
    },
    "stylish": {
        "label": "stylish & visual",
        "genres": ["Science Fiction", "Crime", "Fantasy"],
        "vibes": ["stylish", "slick", "neon-noir", "glamorous", "post-apocalyptic", "surreal", "moody"],
        "keywords": ["heist", "period"],
    },
    "clever": {
        "label": "clever & cerebral",
        "genres": ["Mystery", "Thriller", "Drama", "Science Fiction"],
        "vibes": ["clever", "cerebral", "dialogue-driven", "suspenseful"],
        "keywords": ["mockumentary", "satirical"],
    },
    "bittersweet": {
        "label": "bittersweet",
        "genres": ["Drama", "Romance"],
        "vibes": ["bittersweet", "melancholy", "reflective", "nostalgic", "quiet", "bittersweet"],
        "keywords": ["coming-of-age", "slice of life"],
    },
    "nostalgic": {
        "label": "nostalgic",
        "genres": ["Comedy", "Drama", "Documentary"],
        "vibes": ["nostalgic", "retro", "classic", "warm", "gentle"],
        "keywords": ["classic", "period"],
    },
    "melancholy": {
        "label": "melancholy",
        "genres": ["Drama"],
        "vibes": ["melancholy", "quiet", "bleak", "bittersweet", "reflective", "melancholy"],
        "keywords": [],
    },
    "weird": {
        "label": "weird & offbeat",
        "genres": ["Comedy", "Science Fiction", "Mystery"],
        "vibes": ["weird", "quirky", "surreal", "deadpan", "cerebral", "unconventional", "twisty"],
        "keywords": ["quirky", "surreal"],
    },
    "spooky": {
        "label": "spooky & eerie",
        "genres": ["Horror", "Mystery"],
        "vibes": ["spooky", "creepy", "gothic", "unsettling", "quiet", "supernatural", "mystical"],
        "keywords": ["folk-horror", "gothic", "goblin"],
    },
    "realistic": {
        "label": "grounded & realistic",
        "genres": ["Drama", "Documentary"],
        "vibes": ["realistic", "gritty", "reflective", "slow-burn", "grounded"],
        "keywords": ["documentary", "biopic"],
    },
    "family": {
        "label": "family-friendly",
        "genres": ["Family", "Animation", "Comedy", "Adventure"],
        "vibes": ["family", "cozy", "fun", "warm", "uplifting", "whimsical"],
        "keywords": ["animation", "kids"],
        "avoid": ["gory", "brutal", "dark"],
    },
    "brainy": {
        "label": "smart & rewarding",
        "genres": ["Science Fiction", "Mystery", "Drama"],
        "vibes": ["cerebral", "clever", "character-study", "gritty", "patient", "smart"],
        "keywords": ["mockumentary"],
    },
    "camp": {
        "label": "campy & over the top",
        "genres": ["Action", "Horror", "Comedy"],
        "vibes": ["campy", "stylized", "savage", "spectacle", "noisy", "dark-funny"],
        "keywords": ["camp", "spoof"],
    },
    "indie": {
        "label": "indie & arthouse",
        "genres": ["Drama", "Mystery"],
        "vibes": ["festival-prestige", "stylish", "character-study", "surreal", "bittersweet"],
        "keywords": ["arthouse", "festival"],
    },
}

MOOD_ALIASES: dict[str, str] = {
    "lighthearted": "light", "light-hearted": "light", "light & easy": "light",
    "easy": "light", "breezy": "light", "easygoing": "light", "easy-going": "light",
    "hilarious": "funny", "comedic": "funny", "comedy": "funny", "goofy": "funny",
    "uplifting": "feel-good", "feelgood": "feel-good", "happy": "feel-good",
    "wholesome": "feel-good", "uplift": "feel-good", "positive": "feel-good",
    "comforting": "cozy", "comfort": "cozy", "chill": "cozy", "laid-back": "cozy",
    "comfortable": "cozy", "easy watching": "cozy", "easy watch": "cozy",
    "soothing": "cozy", "relaxing": "cozy",
    "low effort": "light", "unwind": "cozy", "wind down": "cozy",
    "background": "light", "something easy": "light", "doesn't matter what": "light",
    "heartfelt": "emotional", "moving": "emotional", "touching": "emotional",
    "cry": "emotional", "tearjerker": "emotional", "weepy": "emotional",
    "love": "romantic", "rom com": "romantic", "date night": "romantic", "flirty": "romantic",
    "inspiration": "inspiring", "motivational": "inspiring", "feel good true story": "inspiring",
    "suspense": "thrilling", "suspenseful": "thrilling", "gripping": "thrilling",
    "page turner": "thrilling", "tense": "thrilling", "exciting": "action",
    "plot twist": "twisty", "twist": "twisty", "mystery": "twisty", "puzzle": "twisty",
    "mind-bending": "twisty", "mind bending": "twisty", "mindbending": "twisty",
    "mind-blowing": "twisty", "high concept": "twisty", "trippy": "twisty",
    "psychological": "dark", "gritty": "dark", "edgy": "dark", "morbid": "dark",
    "cerebral": "clever", "smart": "clever", "intellectual": "clever",
    "thought provoking": "brainy", "satisfying": "brainy",
    "horror": "scary", "frightening": "scary", "fright": "scary", "terrifying": "scary",
    "creepy": "spooky", "eerie": "spooky", "haunting": "spooky", "ghost story": "spooky",
    "hard": "intense", "heavy": "intense", "gripping violence": "intense",
    "adrenaline": "action", "blockbuster": "action", "mass masala": "action",
    "adventure": "adventurous", "quest": "adventurous", "journey": "adventurous",
    "visual": "stylish", "glamorous": "stylish", "slick": "stylish", "cinematography": "stylish",
    "melancholic": "melancholy", "sad": "melancholy", "sombre": "melancholy", "somber": "melancholy",
    "quirky": "weird", "offbeat": "weird", "unusual": "weird", "strange": "weird",
    "fun": "funny", "kids": "family", "family friendly": "family", "family-friendly": "family",
    "watch with family": "family", "arthouse": "indie", "art house": "indie",
    "suspense": "thrilling", "mystery": "twisty", "noir": "dark", "psychological": "dark",
    "campy": "camp", "over the top": "camp", "bollywood masala": "camp",
}

TYPE_ALIASES: dict[str, str] = {
    "movie": "movie", "film": "movie", "films": "movie", "films?": "movie",
    "cinema": "movie", "feature": "movie", "flick": "movie",
    "show": "series", "series": "series", "tv": "series", "tv show": "series",
    "tv series": "series", "web series": "series", "series?": "series", "anime": "series",
}

# ----------------------------------------------------------------------------- #

_STOPWORDS = {
    # articles, prepositions, pronouns
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "there", "here",
    "me", "my", "i", "you", "your", "we", "our", "they", "them", "their",
    "he", "she", "his", "her", "him", "hers", "do", "does", "did", "have",
    "has", "had", "can", "could", "should", "would", "will", "shall", "may",
    "might", "must", "am", "s", "t", "don", "doesn", "isn", "aren", "wasn",
    # chat filler that carries no search signal
    "about", "what", "whats", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "give", "get", "find", "show", "tell", "know", "think",
    "like", "want", "need", "please", "thanks", "thank", "hi", "hello", "hey",
    "yes", "yeah", "ok", "okay", "sure", "maybe", "just", "really", "very",
    "quite", "much", "many", "lot", "bit", "kind", "sort", "guys", "guy",
    "one", "two", "three", "also", "then", "than", "because", "so", "as",
    "some", "something", "anything", "nothing", "everything", "someone",
    "anyone", "anything", "ever", "always", "never", "again", "back", "even",
    "still", "too", "very", "let", "lets", "make", "made", "take", "put",
    "hiya", "hiy",
    # entertainment nouns: the tools handle these axes explicitly
    "movie", "movies", "film", "films", "cinema", "show", "shows", "series",
    "title", "titles", "thing", "things", "stuff", "watch", "watching",
    "watched", "seen", "see", "binge", "recommend", "recommended",
    "recommends", "recommendation", "recommendations", "suggest",
    "suggested", "suggestion", "suggestions", "pick", "picks", "options",
    "good", "great", "best", "nice", "cool", "awesome", "amazing", "bad",
    "new", "old", "any", "all", "both", "each", "few", "other", "others",
    "another", "same", "different", "under", "less", "over", "above", "below",
    "between", "within", "around", "about", "after", "before", "during",
    "tonight", "today", "tomorrow", "lately", "recently", "currently",
}


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #

def clean(value: Any) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    """Content words only. Short leftovers are dropped too - 'a', 'ok', 'hm'."""
    return [
        t
        for t in re.split(r"[^a-z0-9+]+", clean(text))
        if t and t not in _STOPWORDS and len(t) > 2
    ]


# --------------------------------------------------------------------------- #
# Canonicalisation
# --------------------------------------------------------------------------- #

def canon_genre(value: Any) -> str | None:
    """Map a free-form genre to a canonical one, or None.

    Exact alias match first, then a compound match: every word of the input must
    itself be a known genre, so "romantic comedy" resolves but "quantum musicals"
    does not. A loose substring search is deliberately *not* used - it would map
    any invented genre onto whatever word it happens to contain.
    """
    key = clean(value)
    if not key:
        return None
    if key in GENRE_ALIASES:
        return GENRE_ALIASES[key]

    words = [w for w in re.split(r"\s+", key) if w and w not in _STOPWORDS]
    if not words:
        words = key.split()
    if len(words) > 1 and all(w in GENRE_ALIASES for w in words):
        return GENRE_ALIASES[words[0]]
    return None


def canon_language(value: Any) -> str | None:
    key = clean(value)
    if not key:
        return None
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    for alias, canonical in LANGUAGE_ALIASES.items():
        if len(alias) > 2 and alias in key:
            return canonical
    return None


def canon_mood(value: Any) -> str | None:
    key = clean(value)
    if not key:
        return None
    # "feel good" and "feel-good" are the same mood; try both separators.
    hyphenless = key.replace("-", " ").replace("_", " ")
    for candidate in (key, hyphenless, hyphenless.replace(" ", "-")):
        if candidate in MOOD_MAP:
            return candidate
        if candidate in MOOD_ALIASES:
            return MOOD_ALIASES[candidate]
    # Loose passes use word boundaries, so a short cue like "cry" still matches
    # "i want to cry" without "fun" matching "fundamental".
    for alias, canonical in MOOD_ALIASES.items():
        if len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", hyphenless):
            return canonical
    for alias, canonical in MOOD_ALIASES.items():
        flat = canonical.replace("-", " ")
        if len(flat) > 2 and re.search(rf"\b{re.escape(flat)}\b", hyphenless):
            return canonical
    return None


def canon_type(value: Any) -> str | None:
    key = clean(value)
    if not key:
        return None
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    for alias, canonical in TYPE_ALIASES.items():
        if alias in key:
            return canonical
    return None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _normalise_record(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """Map any source record into CineAgent's single schema."""
    genres = raw.get("genres") or []
    if isinstance(genres, str):
        genres = [genres]
    genres = [canon_genre(g) or str(g).strip() for g in genres if g]

    runtime = raw.get("runtime_minutes") or raw.get("runtime")
    try:
        runtime = int(runtime) if runtime not in (None, "", 0) else None
    except (TypeError, ValueError):
        runtime = None

    seasons = raw.get("seasons")
    try:
        seasons = int(seasons) if seasons not in (None, "") else None
    except (TypeError, ValueError):
        seasons = None

    kind = canon_type(raw.get("type") or raw.get("media_type")) or "movie"

    return {
        "id": str(raw.get("id")),
        "title": str(raw.get("title") or raw.get("name") or "Untitled").strip(),
        "year": int(raw["year"]) if str(raw.get("year", "")).strip("-").isdigit() else None,
        "type": kind,
        "genres": genres,
        "language": canon_language(raw.get("language") or raw.get("original_language")) or "English",
        "runtime_minutes": runtime,
        "seasons": seasons,
        "overview": (raw.get("overview") or "").strip(),
        "director": (raw.get("director") or raw.get("creator") or "").strip(),
        "cast": raw.get("cast") or [],
        "rating": float(raw["rating"]) if raw.get("rating") not in (None, "") else None,
        "maturity": (raw.get("maturity") or raw.get("certification") or "").strip(),
        "vibes": [clean(v) for v in (raw.get("vibes") or []) if v],
        "source": source,
    }


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    """Return the full catalog, sorted by popularity (rating desc)."""
    records: list[dict[str, Any]] = []

    if not LOCAL_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found at {LOCAL_DATASET}")

    with LOCAL_DATASET.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    for raw in payload.get("titles", []):
        records.append(_normalise_record(raw, "local"))

    records.sort(key=lambda r: (-(r["rating"] or 0), r["title"]))
    return records


@lru_cache(maxsize=1)
def catalog_facets() -> dict[str, Any]:
    """Distinct values the LLM can browse, for grounding its tool arguments."""
    titles = load_catalog()
    genres: dict[str, int] = {}
    languages: dict[str, int] = {}
    for t in titles:
        for g in t["genres"]:
            genres[g] = genres.get(g, 0) + 1
        languages[t["language"]] = languages.get(t["language"], 0) + 1
    runtimes = [t["runtime_minutes"] for t in titles if t["runtime_minutes"]]
    return {
        "total": len(titles),
        "genres": sorted(genres, key=lambda g: -genres[g]),
        "languages": sorted(languages, key=lambda l: -languages[l]),
        "moods": sorted(MOOD_MAP),
        "types": ["movie", "series"],
        "runtime_range": [min(runtimes), max(runtimes)] if runtimes else [0, 0],
        "top_rated": [t["title"] for t in titles[:5]],
    }


# --------------------------------------------------------------------------- #
# Matching / scoring
# --------------------------------------------------------------------------- #

def _haystack(t: dict[str, Any]) -> str:
    """Everything we would let free-text search touch, lowercased."""
    bits = [t["title"], t["language"], " ".join(t["genres"]), " ".join(t["vibes"]), t["overview"]]
    if t["director"]:
        bits.append(t["director"])
    bits.extend(str(c) for c in t["cast"][:4])
    return clean(" ".join(b for b in bits if b))


def title_matches(t: dict[str, Any], phrase: str) -> bool:
    """Case/accident-insensitive title containment.

    The reverse direction (catalog title contained in the user's phrase) is only
    trusted for titles of 3+ characters. Without that guard, a phrase like
    "not a real film" matches the title "I", because "i" is a substring of it.
    """
    needle = clean(phrase)
    hay = clean(t["title"])
    if not needle or not hay:
        return False
    if needle == hay or needle in hay:
        return True
    if len(hay) >= 3 and len(needle) >= 3 and hay in needle:
        return True
    # tolerate "the" prefix differences
    stripped = re.sub(r"^(the|a|an)\s+", "", hay)
    return needle == stripped or needle in stripped


def resolve_titles(phrases: Iterable[str]) -> list[dict[str, Any]]:
    """Turn a list of title-ish strings into catalog records (best effort)."""
    catalog = load_catalog()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for phrase in phrases or []:
        if not phrase:
            continue
        for t in catalog:
            if t["id"] in seen:
                continue
            if title_matches(t, phrase):
                out.append(t)
                seen.add(t["id"])
                break
    return out


def mood_score(t: dict[str, Any], mood: str) -> float:
    """0..10 affinity between a title and a mood key."""
    spec = MOOD_MAP.get(mood)
    if not spec:
        return 0.0
    score = 0.0
    genres = set(t["genres"])
    for g in spec["genres"]:
        if g in genres:
            score += 3.0
    vibes = set(t["vibes"])
    for v in spec["vibes"]:
        if v in vibes:
            score += 2.5
    blob = _haystack(t)
    for kw in spec["keywords"]:
        if clean(kw) in blob:
            score += 1.0
    for bad in spec.get("avoid", []):
        if bad in vibes or bad in blob:
            score -= 2.0
    # a little help from the description wording, for free-text moods
    for word in spec["label"].split():
        if len(word) > 4 and word in blob:
            score += 0.5
    return round(min(score, 10.0), 2)


def keyword_score(t: dict[str, Any], words: Iterable[str]) -> float:
    """Loose free-text affinity used by get_details / recommend_by_keyword."""
    blob = _haystack(t)
    score = 0.0
    for w in words:
        w = clean(w)
        if len(w) > 2 and w in blob:
            score += 2.0
    return score


def base_relevance(t: dict[str, Any]) -> float:
    """Prior used to rank results when the query is otherwise flat."""
    return (t["rating"] or 0) * 0.35


def rank(titles: list[dict[str, Any]], scores: dict[str, float]) -> list[dict[str, Any]]:
    """Stable sort by score, then rating, then title."""
    return sorted(
        titles,
        key=lambda t: (-scores.get(t["id"], 0.0), -(t["rating"] or 0), t["title"]),
    )
