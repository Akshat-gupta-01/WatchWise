"""
tools.py
========
Every capability the LLM has, declared twice: once as a Python function, once as
a JSON schema the model can call.

Design notes
------------
* **One source of truth.** `_TOOL_SPECS` holds name, description, JSON schema and
  the Python callable. `execute()` dispatches by name. Adding a tool = adding one
  dict entry plus one function.
* **Every tool accepts `restrict_to`.** This is the trick that makes follow-up
  questions work without re-running the whole conversation: the model passes the
  titles it mentioned last turn and the tool narrows *within* them. "Any of those
  with more comedy?" -> search_by_genre("comedy", restrict_to=[...]).
* **Tools return compact records**, not the whole catalog, so the context window
  survives. `brief()` is the single place that decides what the model sees.
* **Forgiving arguments.** The model says "bollywood", "sci fi", "short" - we
  canonicalise before touching data. Every tool also returns a human-readable
  `note` when it had to reinterpret something, so the model can mention it.
"""

from __future__ import annotations

from typing import Any, Callable

import catalog as C

MAX_RESULTS = 12
DEFAULT_RESULTS = 5

# --------------------------------------------------------------------------- #
# Output shaping
# --------------------------------------------------------------------------- #


def brief(t: dict[str, Any]) -> dict[str, Any]:
    """The compact projection the LLM actually reads."""
    out = {
        "title": t["title"],
        "year": t["year"],
        "type": t["type"],
        "genres": t["genres"][:3],
        "language": t["language"],
        "runtime_minutes": t["runtime_minutes"],
        "rating": t["rating"],
        "maturity": t["maturity"],
        "hook": t["overview"][:180],
    }
    if t["type"] == "series":
        out["seasons"] = t["seasons"]
    return out


def result(
    titles: list[dict[str, Any]],
    summary: str,
    *,
    notes: list[str] | None = None,
    limit: int = DEFAULT_RESULTS,
) -> dict[str, Any]:
    shown = titles[:limit]
    return {
        "ok": True,
        "summary": summary,
        "count": len(shown),
        "total_matches": len(titles),
        "notes": notes or [],
        "titles": [brief(t) for t in shown],
    }


def empty(summary: str, note: str, *, limit: int = DEFAULT_RESULTS) -> dict[str, Any]:
    return {
        "ok": True,
        "summary": summary,
        "count": 0,
        "total_matches": 0,
        "notes": [note],
        "titles": [],
    }


# --------------------------------------------------------------------------- #
# Shared filter plumbing
# --------------------------------------------------------------------------- #

def _clamp_limit(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RESULTS
    return max(1, min(n, MAX_RESULTS))


def _restrict(
    args: dict[str, Any], notes: list[str]
) -> list[dict[str, Any]] | None:
    """Resolve an optional `restrict_to` list of titles into records."""
    phrases = args.get("restrict_to") or []
    if not phrases:
        return None
    if isinstance(phrases, str):
        phrases = [phrases]
    titles = C.resolve_titles(phrases)
    if not titles:
        notes.append(
            f"Could not match restrict_to {list(phrases)!r} to the catalog; "
            "searched the full catalog instead."
        )
        return None
    notes.append(f"Searched only within: {', '.join(t['title'] for t in titles)}.")
    return titles


def _type_filter(
    args: dict[str, Any], notes: list[str]
) -> Callable[[dict[str, Any]], bool] | None:
    if not args.get("type"):
        return None
    kind = C.canon_type(args.get("type") or args.get("media_type"))
    if not kind:
        return None
    notes.append(f"Type filter applied: {kind} only.")
    return lambda t: t["type"] == kind


# --------------------------------------------------------------------------- #
# Tool 1: genre
# --------------------------------------------------------------------------- #

def search_by_genre(
    genre: str,
    limit: int = DEFAULT_RESULTS,
    restrict_to: list[str] | None = None,
    type: str | None = None,
) -> dict[str, Any]:
    """Best titles in a genre (or several), optionally narrowed to titles you already mentioned."""
    notes: list[str] = []
    canonical = C.canon_genre(genre)
    if not canonical:
        available = ", ".join(C.catalog_facets()["genres"])
        return empty(
            f"I do not know a genre called {genre!r}.",
            f"Available genres: {available}. Pick one of these and call again.",
        )
    if canonical != C.clean(genre).title():
        notes.append(f"Read {genre!r} as the genre '{canonical}'.")

    pool = _restrict({"restrict_to": restrict_to}, notes) or C.load_catalog()
    kind_filter = _type_filter({"type": type}, notes)

    wanted = {canonical}
    for extra in (genre if isinstance(genre, list) else [genre]):
        c = C.canon_genre(extra)
        if c and c != canonical:
            wanted.add(c)

    hits, scores = [], {}
    for t in pool:
        overlap = wanted & set(t["genres"])
        if not overlap:
            # loose pass: genre mentioned in the blurb
            blob = C.clean(t["overview"] + " " + " ".join(t["vibes"]))
            if any(C.clean(w) in blob for w in wanted):
                overlap = wanted
        if not overlap:
            continue
        if kind_filter and not kind_filter(t):
            continue
        score = 4.0 * len(overlap) + C.base_relevance(t)
        hits.append(t)
        scores[t["id"]] = score

    ranked = C.rank(hits, scores)
    if not ranked:
        return empty(
            f"No {canonical} titles in the catalog.",
            "Widen the search - drop the type filter or try another genre.",
        )
    genre_list = " or ".join(sorted(wanted))
    prefix = (
        f"Among the titles you mentioned, here are the {genre_list} picks"
        if restrict_to
        else f"Here are the top {genre_list} picks"
    )
    return result(ranked, f"{prefix}.", notes=notes, limit=_clamp_limit(limit))


# --------------------------------------------------------------------------- #
# Tool 2: mood
# --------------------------------------------------------------------------- #

def search_by_mood(
    mood: str,
    limit: int = DEFAULT_RESULTS,
    restrict_to: list[str] | None = None,
    type: str | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    """Match a *feeling* - 'light', 'cozy', 'mind-bending', 'comforting' - not just a genre."""
    notes: list[str] = []
    key = C.canon_mood(mood)
    if not key:
        known = ", ".join(C.MOOD_MAP)
        return empty(
            f"I do not have a read on the mood {mood!r}.",
            f"Supported moods: {known}. Call again with one of these.",
        )
    spec = C.MOOD_MAP[key]
    if C.clean(mood) != key:
        notes.append(f"Read {mood!r} as the mood '{key}' ({spec['label']}).")

    pool = _restrict({"restrict_to": restrict_to}, notes) or C.load_catalog()
    kind_filter = _type_filter({"type": type}, notes)
    floor = float(min_rating) if isinstance(min_rating, (int, float)) else 0.0

    hits, scores = [], {}
    for t in pool:
        if kind_filter and not kind_filter(t):
            continue
        if floor and (t["rating"] or 0) < floor:
            continue
        s = C.mood_score(t, key)
        if s <= 0:
            continue
        hits.append(t)
        scores[t["id"]] = s + C.base_relevance(t) * 0.3

    ranked = C.rank(hits, scores)
    if not ranked:
        return empty(
            f"Nothing in the catalog reads as '{spec['label']}'.",
            "Try a neighbouring mood or drop the type filter.",
        )
    if restrict_to:
        head = f"Of the titles you mentioned, these lean most '{spec['label']}'"
    else:
        head = f"Here is what reads as '{spec['label']}'"
    return result(ranked, f"{head}.", notes=notes, limit=_clamp_limit(limit))


# --------------------------------------------------------------------------- #
# Tool 3: runtime
# --------------------------------------------------------------------------- #

def filter_by_runtime(
    max_minutes: int,
    min_minutes: int | None = None,
    limit: int = DEFAULT_RESULTS,
    restrict_to: list[str] | None = None,
    type: str | None = None,
    sort_by: str = "rating",
) -> dict[str, Any]:
    """Keep titles at or under a runtime. For series this is the per-episode runtime."""
    notes: list[str] = []
    try:
        hi = int(max_minutes)
    except (TypeError, ValueError):
        return empty(
            f"Could not read {max_minutes!r} as a number of minutes.",
            "Pass max_minutes as an integer, e.g. 90.",
        )
    lo = int(min_minutes) if isinstance(min_minutes, (int, float)) else 0
    if lo > hi:
        notes.append(f"min_minutes {lo} exceeded max_minutes {hi}; searching under {hi} only.")
        lo = 0

    pool = _restrict({"restrict_to": restrict_to}, notes) or C.load_catalog()
    kind_filter = _type_filter({"type": type}, notes)

    hits, scores = [], {}
    for t in pool:
        rt = t["runtime_minutes"]
        if not rt or rt > hi or rt < lo:
            continue
        if kind_filter and not kind_filter(t):
            continue
        hits.append(t)
        scores[t["id"]] = C.base_relevance(t)

    if C.clean(sort_by) in ("shortest", "runtime", "length"):
        # "give me the quickest thing under 90 minutes"
        ranked = sorted(hits, key=lambda t: (t["runtime_minutes"] or 9999, -(t["rating"] or 0)))
    else:
        # "give me the best thing under 90 minutes" - best first, length only breaks ties
        ranked = sorted(
            hits,
            key=lambda t: (-(t["rating"] or 0), t["runtime_minutes"] or 9999),
        )

    if not ranked:
        return empty(
            f"Nothing in the catalog runs between {lo} and {hi} minutes.",
            "Widen max_minutes or drop the type filter.",
        )
    window = f"under {hi} min" if lo == 0 else f"between {lo} and {hi} min"
    if restrict_to:
        head = f"Of the titles you mentioned, these fit {window}"
    else:
        head = f"Here is what fits {window}"
    return result(ranked, f"{head}.", notes=notes, limit=_clamp_limit(limit))


# --------------------------------------------------------------------------- #
# Tool 4: language
# --------------------------------------------------------------------------- #

def filter_by_language(
    lang: str,
    limit: int = DEFAULT_RESULTS,
    restrict_to: list[str] | None = None,
    type: str | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    """Keep titles in one language. Accepts 'Hindi', 'hi', 'bollywood', 'k-drama'."""
    notes: list[str] = []
    canonical = C.canon_language(lang)
    if not canonical:
        known = ", ".join(C.catalog_facets()["languages"])
        return empty(
            f"The catalog has no {lang!r} titles.",
            f"Available languages: {known}. Call again with one of these.",
        )
    if C.clean(lang) != canonical.lower():
        notes.append(f"Read {lang!r} as the language '{canonical}'.")

    pool = _restrict({"restrict_to": restrict_to}, notes) or C.load_catalog()
    kind_filter = _type_filter({"type": type}, notes)
    floor = float(min_rating) if isinstance(min_rating, (int, float)) else 0.0

    hits, scores = [], {}
    for t in pool:
        if t["language"] != canonical:
            continue
        if kind_filter and not kind_filter(t):
            continue
        if floor and (t["rating"] or 0) < floor:
            continue
        hits.append(t)
        scores[t["id"]] = C.base_relevance(t)

    ranked = C.rank(hits, scores)
    if not ranked:
        return empty(
            f"No {canonical} titles in the catalog.",
            "Try a different language or drop the type filter.",
        )
    if restrict_to:
        head = f"Of the titles you mentioned, these are the {canonical} ones"
    else:
        head = f"Here are the top {canonical} picks"
    return result(ranked, f"{head}.", notes=notes, limit=_clamp_limit(limit))


# --------------------------------------------------------------------------- #
# Tool 5: composed multi-constraint search
# --------------------------------------------------------------------------- #

def recommend_titles(
    mood: str | None = None,
    genre: str | None = None,
    lang: str | None = None,
    max_minutes: int | None = None,
    min_minutes: int | None = None,
    type: str | None = None,
    min_rating: float | None = None,
    exclude: list[str] | None = None,
    restrict_to: list[str] | None = None,
    limit: int = DEFAULT_RESULTS,
) -> dict[str, Any]:
    """Search every axis in one pass, so all constraints must hold at the same time.

    Use this whenever the request carries two or more constraints ("something
    light, under 90 minutes, in Hindi"). It is exact: a returned title satisfies
    every constraint you passed, rather than the overlap of separate result sets.
    """
    notes: list[str] = []
    want_mood = C.canon_mood(mood) if mood else None
    want_genre = C.canon_genre(genre) if genre else None
    want_lang = C.canon_language(lang) if lang else None
    want_type = C.canon_type(type) if type else None

    if mood and not want_mood:
        notes.append(f"Ignored unknown mood {mood!r}; supported: {', '.join(C.MOOD_MAP)}.")
    if genre and not want_genre:
        notes.append(f"Ignored unknown genre {genre!r}; use list_catalog_facets to check names.")
    if lang and not want_lang:
        notes.append(f"Ignored unknown language {lang!r}; use list_catalog_facets to check names.")

    hi = int(max_minutes) if isinstance(max_minutes, (int, float)) else None
    lo = int(min_minutes) if isinstance(min_minutes, (int, float)) else None
    floor = float(min_rating) if isinstance(min_rating, (int, float)) else 0.0
    banned = {C.clean(x) for x in (exclude or [])}

    pool = C.load_catalog()
    if restrict_to:
        narrowed = C.resolve_titles(restrict_to)
        if narrowed:
            notes.append(f"Searched only within: {', '.join(t['title'] for t in narrowed)}.")
            pool = narrowed
        else:
            notes.append("Could not match restrict_to to the catalog; searched everything.")

    hits, scores = [], {}
    for t in pool:
        if banned and C.clean(t["title"]) in banned:
            continue
        if want_lang and t["language"] != want_lang:
            continue
        if want_type and t["type"] != want_type:
            continue
        if floor and (t["rating"] or 0) < floor:
            continue
        rt = t["runtime_minutes"]
        if hi is not None:
            if not rt or rt > hi:
                continue
        if lo is not None and (not rt or rt < lo):
            continue
        if want_genre and want_genre not in t["genres"]:
            continue

        score = 0.0
        if want_mood:
            mood_hit = C.mood_score(t, want_mood)
            if mood_hit <= 0:
                continue
            score += mood_hit * 2.0
        if want_genre:
            score += 4.0
        score += C.base_relevance(t)
        if hi is not None and rt:
            score += 2.0 * (1 - rt / hi)  # reward using the budget well, not just fitting

        hits.append(t)
        scores[t["id"]] = score

    ranked = C.rank(hits, scores)
    if not ranked:
        return empty(
            "Nothing in the catalog satisfies all of those constraints at once.",
            "Relax one filter - for example raise the runtime ceiling or drop the language.",
        )

    stated: list[str] = []
    if want_mood:
        stated.append(f"{C.MOOD_MAP[want_mood]['label']}")
    if want_genre:
        stated.append(want_genre)
    if want_lang:
        stated.append(want_lang)
    if hi is not None:
        stated.append(f"under {hi} min")
    if want_type:
        stated.append(want_type)
    return result(
        ranked,
        f"Titles that satisfy every constraint at once: {', '.join(stated) or 'your request'}.",
        notes=notes,
        limit=_clamp_limit(limit),
    )


# --------------------------------------------------------------------------- #
# Support tools (grounding + follow-ups)
# --------------------------------------------------------------------------- #

def list_catalog_facets() -> dict[str, Any]:
    """What is actually in the catalog: genres, languages, moods, runtime range."""
    f = C.catalog_facets()
    return {
        "ok": True,
        "summary": (
            f"{f['total']} titles. Genres: {', '.join(f['genres'])}. "
            f"Languages: {', '.join(f['languages'])}. "
            f"Runtimes {f['runtime_range'][0]}-{f['runtime_range'][1]} min. "
            f"Highest rated: {', '.join(f['top_rated'])}."
        ),
        "genres": f["genres"],
        "languages": f["languages"],
        "moods": f["moods"],
        "types": f["types"],
        "runtime_range_minutes": f["runtime_range"],
        "notes": ["Moods are the keyword for search_by_mood; every other axis has its own tool."],
    }


def get_title_details(title: str) -> dict[str, Any]:
    """Full detail for one title, for when the user asks 'what is X about?'."""
    matches = C.resolve_titles([title])
    if not matches:
        return {
            "ok": False,
            "summary": f"No catalog title matched {title!r}.",
            "count": 0,
            "notes": ["Use search_titles to find the closest match first."],
            "titles": [],
        }
    t = matches[0]
    return {
        "ok": True,
        "summary": f"{t['title']} ({t['year']}) - {t['type']}, {t['language']}.",
        "count": 1,
        "total_matches": 1,
        "notes": [],
        "titles": [
            {
                **brief(t),
                "full_overview": t["overview"],
                "vibes": t["vibes"],
                "director": t["director"],
                "cast": t["cast"],
                "seasons": t["seasons"],
                "id": t["id"],
            }
        ],
    }


def search_titles(
    query: str,
    limit: int = DEFAULT_RESULTS,
    type: str | None = None,
) -> dict[str, Any]:
    """Free-text search across title, genre, vibe, cast, director and synopsis."""
    notes: list[str] = []
    words = C.tokens(query)
    if not words:
        return empty(
            "That query had nothing searchable in it.",
            "Try a title, an actor, a director, or a genre.",
        )
    kind_filter = _type_filter({"type": type}, notes)

    hits, scores = [], {}
    for t in C.load_catalog():
        if kind_filter and not kind_filter(t):
            continue
        s = C.keyword_score(t, words)
        if s <= 0:
            continue
        # exact title phrase is the strongest signal there is
        if C.title_matches(t, query):
            s += 20
        hits.append(t)
        scores[t["id"]] = s + C.base_relevance(t)

    ranked = C.rank(hits, scores)
    if not ranked:
        return empty(
            f"Nothing matched {query!r}.",
            "Try fewer words, or use search_by_mood / search_by_genre instead.",
        )
    return result(
        ranked,
        f"Closest matches for {query!r}.",
        notes=notes,
        limit=_clamp_limit(limit),
    )


def recommend_similar(
    title: str,
    limit: int = DEFAULT_RESULTS,
    exclude_current: bool = True,
) -> dict[str, Any]:
    """Titles that feel like a given one: shared genre, language, vibe and era."""
    notes: list[str] = []
    matches = C.resolve_titles([title])
    if not matches:
        return empty(
            f"No catalog title matched {title!r} for similarity.",
            "Use search_titles to find the closest match first.",
        )
    seed = matches[0]
    seed_genres, seed_vibes = set(seed["genres"]), set(seed["vibes"])

    hits, scores = [], {}
    for t in C.load_catalog():
        if exclude_current and t["id"] == seed["id"]:
            continue
        s = 0.0
        s += 4.0 * len(seed_genres & set(t["genres"]))
        s += 1.5 * len(seed_vibes & set(t["vibes"]))
        if t["language"] == seed["language"]:
            s += 2.0
        if seed["year"] and t["year"]:
            gap = abs(t["year"] - seed["year"])
            s += 2.0 if gap <= 6 else (1.0 if gap <= 15 else 0.0)
        if t["rating"] and seed["rating"]:
            s += 1.0 - abs(t["rating"] - seed["rating"]) / 5
        if s <= 3:
            continue
        hits.append(t)
        scores[t["id"]] = s

    ranked = C.rank(hits, scores)
    if not ranked:
        return empty(
            f"Nothing in the catalog is close to {seed['title']}.",
            "Try a broader seed title.",
        )
    return result(
        ranked,
        f"If you liked {seed['title']} ({seed['year']}), try these.",
        notes=notes,
        limit=_clamp_limit(limit),
    )


# --------------------------------------------------------------------------- #
# Declarative spec: one entry per tool
# --------------------------------------------------------------------------- #

class _Schema:
    """Tiny helper so the specs below read like JSON Schema without the noise."""

    def __init__(self, type_: str, description: str, **props: Any) -> None:
        self.type = type_
        self.description = description
        self.props = props

    def build(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.props:
            # Nested _Schema instances must become plain dicts, or json.dumps
            # in the HTTP client blows up at request time rather than here.
            schema["properties"] = {k: _plain(v) for k, v in self.props.items()}
        return schema


def _plain(value: Any) -> Any:
    """Recursively convert _Schema instances into JSON-serialisable dicts."""
    if isinstance(value, _Schema):
        return value.build()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


_RESTRICT_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "Optional. Titles the assistant already mentioned this conversation, used to "
        "narrow the search to just those. Set this for follow-up questions like "
        "'any of those with more comedy?' instead of re-searching the whole catalog."
    ),
}

_TYPE_SCHEMA = {
    "type": "string",
    "enum": ["movie", "series"],
    "description": "Optional. Restrict to films or TV series.",
}

_LIMIT_SCHEMA = {
    "type": "integer",
    "minimum": 1,
    "maximum": MAX_RESULTS,
    "description": f"How many titles to return. Default 5, maximum {MAX_RESULTS}. "
    "Pass the maximum when you plan to narrow the result set further.",
}

_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_by_genre",
        "description": (
            "Find titles by genre. Call this when the user names a genre (comedy, "
            "thriller, romance, sci-fi, crime, horror, animation...). Returns the "
            "highest rated titles in that genre. For series, runtime_minutes is the "
            "per-episode runtime."
        ),
        "parameters": _Schema(
            "object",
            "Search the catalog by genre.",
            genre=_Schema(
                "string",
                "The genre, e.g. 'comedy', 'psychological thriller', 'sci-fi'.",
            ),
            limit=_LIMIT_SCHEMA,
            restrict_to=_RESTRICT_SCHEMA,
            type=_TYPE_SCHEMA,
        ).build(),
        "fn": search_by_genre,
    },
    {
        "name": "search_by_mood",
        "description": (
            "Find titles by FEELING rather than genre. Use this whenever the user "
            "describes a vibe instead of a category: 'something light', 'cozy for a "
            "rainy evening', 'mind-bending', 'I want to cry', 'weird but interesting', "
            "'comforting', 'not too heavy'. Returns titles scored against that mood."
        ),
        "parameters": _Schema(
            "object",
            "Search the catalog by mood / vibe.",
            mood=_Schema(
                "string",
                "The mood, e.g. 'light', 'feel-good', 'thrilling', 'twisty', 'dark', "
                "'scary', 'cozy', 'romantic', 'emotional', 'inspiring', 'bittersweet', "
                "'stylish', 'weird', 'nostalgic', 'brainy'.",
            ),
            limit=_LIMIT_SCHEMA,
            restrict_to=_RESTRICT_SCHEMA,
            type=_TYPE_SCHEMA,
            min_rating={
                "type": "number",
                "description": "Optional floor on the 0-10 rating, e.g. 7.5.",
            },
        ).build(),
        "fn": search_by_mood,
    },
    {
        "name": "filter_by_runtime",
        "description": (
            "Filter by length. Use whenever the user mentions a time constraint: "
            "'under 90 minutes', 'short', 'something in 2 hours', 'quick watch'. "
            "Works for series too, where it filters on per-episode runtime."
        ),
        "parameters": _Schema(
            "object",
            "Filter by runtime in minutes.",
            max_minutes=_Schema(
                "integer",
                "Upper bound in minutes, inclusive. Use 999 for 'no limit'.",
            ),
            min_minutes={
                "type": "integer",
                "description": "Optional lower bound in minutes.",
            },
            limit=_LIMIT_SCHEMA,
            restrict_to=_RESTRICT_SCHEMA,
            type=_TYPE_SCHEMA,
            sort_by={
                "type": "string",
                "enum": ["rating", "shortest"],
                "description": "Rank by rating (default) or by runtime ascending.",
            },
        ).build(),
        "fn": filter_by_runtime,
    },
    {
        "name": "filter_by_language",
        "description": (
            "Filter by language. Use when the user asks for something in a specific "
            "language or industry: 'in Hindi', 'Hindi movies', 'a K-drama', 'something "
            "in Korean', 'Hollywood'. Also accepts shorthand: 'hi', 'bollywood', 'en'."
        ),
        "parameters": _Schema(
            "object",
            "Filter by language.",
            lang=_Schema(
                "string",
                "Language name or code, e.g. 'Hindi', 'hi', 'English', 'Korean', "
                "'Tamil', 'Telugu', 'Kannada', 'Malayalam'.",
            ),
            limit=_LIMIT_SCHEMA,
            restrict_to=_RESTRICT_SCHEMA,
            type=_TYPE_SCHEMA,
            min_rating={
                "type": "number",
                "description": "Optional floor on the 0-10 rating.",
            },
        ).build(),
        "fn": filter_by_language,
    },
    {
        "name": "recommend_titles",
        "description": (
            "The all-axes search. Use this whenever the request carries TWO OR MORE "
            "constraints at once, e.g. 'something light, under 90 minutes, in Hindi', "
            "'a scary show in Korean', 'comedy under two hours'. Every argument is "
            "optional and every one you pass must hold simultaneously, so a returned "
            "title is guaranteed to satisfy all of them. Prefer this over chaining "
            "several single-axis tools when the request has more than one constraint."
        ),
        "parameters": _Schema(
            "object",
            "Search the catalog on several axes at once.",
            mood=_Schema(
                "string",
                "Optional mood, e.g. 'light', 'cozy', 'mind-bending', 'emotional'.",
            ),
            genre=_Schema("string", "Optional genre, e.g. 'comedy', 'thriller'."),
            lang=_Schema(
                "string",
                "Optional language, e.g. 'Hindi', 'hi', 'Korean', 'Tamil', 'English'.",
            ),
            max_minutes=_Schema(
                "integer",
                "Optional runtime ceiling in minutes. For series this is per-episode.",
            ),
            min_minutes=_Schema("integer", "Optional runtime floor in minutes."),
            type=_TYPE_SCHEMA,
            min_rating={
                "type": "number",
                "description": "Optional floor on the 0-10 rating, e.g. 7.5.",
            },
            exclude={
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional titles to leave out, e.g. ones already rejected.",
            },
            restrict_to=_RESTRICT_SCHEMA,
            limit=_LIMIT_SCHEMA,
        ).build(),
        "fn": recommend_titles,
    },
    {
        "name": "search_titles",
        "description": (
            "Free-text search over title, genre, cast, director and synopsis. Use when "
            "the user mentions a specific person, a director, an actor, a franchise, or "
            "a phrase that is not a clean genre or mood."
        ),
        "parameters": _Schema(
            "object",
            "Free-text search across the catalog.",
            query=_Schema(
                "string",
                "Words to search for, e.g. 'Christopher Nolan', 'Zoya Akhtar', "
                "'heist', 'Aamir Khan'.",
            ),
            limit=_LIMIT_SCHEMA,
            type=_TYPE_SCHEMA,
        ).build(),
        "fn": search_titles,
    },
    {
        "name": "recommend_similar",
        "description": (
            "Given one title, find titles that feel similar (shared genre, language, "
            "vibe and era). Use for 'what else like X?', 'more like this one'."
        ),
        "parameters": _Schema(
            "object",
            "Find titles similar to a given title.",
            title=_Schema("string", "The title to use as the seed, e.g. 'Andhadhun'."),
            limit=_LIMIT_SCHEMA,
        ).build(),
        "fn": recommend_similar,
    },
    {
        "name": "get_title_details",
        "description": (
            "Get full detail on one title: full synopsis, director, cast, seasons, "
            "vibes, rating. Use when the user asks 'what is X about?' or after "
            "recommending a title they then ask about."
        ),
        "parameters": _Schema(
            "object",
            "Look up one title in detail.",
            title=_Schema("string", "The exact or near-exact title."),
        ).build(),
        "fn": get_title_details,
    },
    {
        "name": "list_catalog_facets",
        "description": (
            "List what the catalog actually contains: every genre, every language, every "
            "supported mood, the runtime range, and the highest rated titles. Call this "
            "when you are unsure which genre or language names are valid, before guessing."
        ),
        "parameters": _Schema("object", "List the catalog's available facets.").build(),
        "fn": list_catalog_facets,
    },
]

REGISTRY: dict[str, dict[str, Any]] = {s["name"]: s for s in _TOOL_SPECS}
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    s["name"]: s["fn"] for s in _TOOL_SPECS
}

CORE_TOOL_NAMES = [
    "search_by_genre",
    "search_by_mood",
    "filter_by_runtime",
    "filter_by_language",
]


# --------------------------------------------------------------------------- #
# Provider-specific schema rendering
# --------------------------------------------------------------------------- #

def openai_tools() -> list[dict[str, Any]]:
    """`tools=[...]` payload for the OpenAI Chat Completions API."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in _TOOL_SPECS
    ]


def anthropic_tools() -> list[dict[str, Any]]:
    """`tools=[...]` payload for the Anthropic Messages API.

    Anthropic nests JSON Schema under `input_schema` and has no `type` wrapper.
    """
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in _TOOL_SPECS
    ]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def execute(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a tool by name. Never raises: a failed tool is a message, not a crash."""
    arguments = arguments or {}
    if name not in TOOL_FUNCTIONS:
        return {
            "ok": False,
            "summary": f"Unknown tool {name!r}.",
            "count": 0,
            "notes": [f"Available tools: {', '.join(TOOL_FUNCTIONS)}."],
            "titles": [],
        }
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except TypeError as exc:
        return {
            "ok": False,
            "summary": f"Bad arguments for {name}: {exc}",
            "count": 0,
            "notes": [f"Expected parameters for {name}: {', '.join(REGISTRY[name]['parameters'].get('properties', {}))}."],
            "titles": [],
        }
    except Exception as exc:  # noqa: BLE001 - tools must never take the app down
        return {
            "ok": False,
            "summary": f"{name} failed: {exc}",
            "count": 0,
            "notes": ["Retry with simpler arguments."],
            "titles": [],
        }
