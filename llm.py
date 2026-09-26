"""
llm.py
=======
Everything that talks to a language model, plus the agent loop that turns tool
calls into an answer.

The loop
--------
    messages -> model -> tool_calls?
                         | yes  -> execute in tools.py -> append results -> model (loop)
                         | no   -> final text -> back to the user

    (capped at MAX_TOOL_ITERATIONS so a confused model cannot spin forever)

Providers
---------
`OpenAIClient` and `AnthropicClient` are thin, dependency-free wrappers over the
HTTP APIs using `requests`, so the project installs with no vendor SDK. Both
speak the same *internal* message format:

    {"role": "user" | "assistant" | "tool", "content": str, ...}

and each one translates that into its own wire format. `HeuristicPlanner` is a
no-API-key fallback that runs the same tools with a rule-based intent parser, so
the app is fully demoable offline and doubles as a smoke test of tools.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests

import catalog as C
import tools as T

MAX_TOOL_ITERATIONS = 5
REQUEST_TIMEOUT = 60
DEFAULT_POOL = 8

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are WatchWise, developed by Akshat Gupta and Shekhar Rana, a warm and opinionated movie and TV discovery assistant for an
OTT-style streaming catalog. You talk like a friend who has watched everything,
not like a search engine.

## Your tools
You have tools that query the catalog. You cannot browse the internet and you
have no memory of anything outside this conversation. ALWAYS call at least one
tool before recommending anything.

- `search_by_mood(mood)` - the user described a *feeling*: "something light",
  "cozy", "I want to cry", "mind-bending", "not too heavy". Prefer this over a
  genre whenever the user talks about mood rather than category.
- `search_by_genre(genre)` - the user named a category: comedy, thriller, romance.
- `filter_by_runtime(max_minutes)` - any time constraint: "under 90 minutes",
  "short", "2 hours or less". For series this filters on per-episode runtime.
- `filter_by_language(lang)` - "in Hindi", "a K-drama", "something in Korean".
- `recommend_titles(...)` - ALL AXES IN ONE CALL. Use this whenever the request
  carries two or more constraints at once, e.g. mood + runtime + language. Every
  argument you pass must hold simultaneously, so the titles it returns genuinely
  satisfy all of them. This is more reliable than chaining several single-axis
  tools and intersecting their results by hand.
- `search_titles(query)` - a person, director, actor, franchise or phrase.
- `recommend_similar(title)` - "what else like X?"
- `get_title_details(title)` - "what is X about?" or a follow-up on a pick.
- `list_catalog_facets()` - check which genres/languages/moods actually exist
  before you guess a name. The catalog is not the whole world.

## How to handle multi-constraint requests
Use ONE `recommend_titles` call for "something light, under 90 minutes, in Hindi".
Use the single-axis tools when there is only one constraint, and when a follow-up
needs a different tool than the one you used before ("lighter?", "in Tamil?").
Never call a tool whose filter you already know will return nothing.

## Follow-up questions
The conversation is continuous. Titles you already recommended are still in play.
When the user says "any of those with more comedy?" or "shorter ones", or "what
about the second one?", do NOT re-search the whole catalog. Pass the titles you
are narrowing to in the `restrict_to` argument of the tool - every search tool
accepts it - and search inside that set. For "the second one" or "tell me more
about that", call `get_title_details` with that one title. Re-recommending titles
the user has already rejected is a mistake.

## Rules
- Never invent a title, actor, director or plot. If the tools did not return it,
  it is not in the catalog. If nothing matches, say so plainly and offer the
  nearest alternative instead of padding the list.
- Recommend 3 to 5 titles. Fewer if the constraints are genuinely narrow, and
  say why it is narrow.
- Reply in Markdown. Open with one short, natural sentence that shows you
  understood the request. Then a bulleted list of 3-5 titles, each as
  **Title (Year)** followed by a single vivid line about why it fits. Then one
  short closing line, ideally a question that offers a natural next step
  ("Want something darker, or more like the second one?").
- Keep it to about 120 words. Warm and specific beats long and generic. No
  markdown tables, no headers, no emoji spam.
- Mention runtime, language and format when they are relevant to the ask, and
  flag anything the user should know about content intensity.
"""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class ToolCallTrace:
    """One tool invocation, surfaced in the UI so the reasoning is visible."""

    name: str
    arguments: dict[str, Any]
    summary: str
    count: int
    titles: list[str] = field(default_factory=list)


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    provider: str = "none"
    model: str = "none"
    used_fallback: bool = False
    note: str = ""
    picks: list[dict[str, Any]] = field(default_factory=list)


class LLMError(RuntimeError):
    """Any failure worth showing the user instead of a stack trace."""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _load_json_arg(raw: Any) -> dict[str, Any]:
    """Models sometimes hand back a JSON string, sometimes a dict, sometimes junk."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _best_picks(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Structured picks for the UI's title cards, taken from the richest tool result.

    Prefers `recommend_titles` (it already enforces every constraint) and falls
    back to the first tool that returned anything.
    """
    if not outputs:
        return []
    for output in outputs:
        if output.get("titles") and output.get("count"):
            return output["titles"]
    return []


def run_tool_calls(
    calls: list[tuple[str, dict[str, Any]]],
) -> tuple[list[ToolCallTrace], list[dict[str, Any]]]:
    """Execute (name, args) pairs, returning both the UI traces and tool payloads."""
    traces: list[ToolCallTrace] = []
    outputs: list[dict[str, Any]] = []
    for name, args in calls:
        result = T.execute(name, args)
        outputs.append(result)
        traces.append(
            ToolCallTrace(
                name=name,
                arguments=args,
                summary=result.get("summary", ""),
                count=int(result.get("count") or 0),
                titles=[t["title"] for t in result.get("titles", [])],
            )
        )
    return traces, outputs


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #

class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # -- one request --------------------------------------------------------- #
    def _post(self, messages: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": T.openai_tools(),
            "tool_choice": "auto",
            "temperature": temperature,
        }
        last_error: str = "unknown error"
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code == 200:
                return resp.json()

            body = resp.text[:400]
            if resp.status_code in (408, 429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}: {body}"
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"OpenAI returned HTTP {resp.status_code}: {body}")

        raise LLMError(f"OpenAI request failed after 3 attempts ({last_error}).")

    # -- the loop ----------------------------------------------------------- #
    def respond(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> AgentReply:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": user_message},
        ]

        traces: list[ToolCallTrace] = []

        picks: list[dict[str, Any]] = []
        for _ in range(MAX_TOOL_ITERATIONS):
            data = self._post(messages, temperature)
            message = data["choices"][0]["message"]
            raw_calls = message.get("tool_calls") or []

            if not raw_calls:
                return AgentReply(
                    text=(message.get("content") or "").strip()
                    or "I could not put together an answer. Could you rephrase that?",
                    tool_calls=traces,
                    provider=self.provider,
                    model=self.model,
                    picks=picks,
                )

            pairs = [
                (c["function"]["name"], _load_json_arg(c["function"].get("arguments")))
                for c in raw_calls
            ]
            new_traces, outputs = run_tool_calls(pairs)
            traces.extend(new_traces)
            picks = _best_picks(outputs) or picks

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": raw_calls,
                }
            )
            for call, output in zip(raw_calls, outputs):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "content": _dump(output),
                    }
                )

        # Ran out of iterations: answer with whatever the tools last returned.
        return AgentReply(
            text=self._synthesise(traces),
            tool_calls=traces,
            provider=self.provider,
            model=self.model,
            note=f"Stopped after {MAX_TOOL_ITERATIONS} tool rounds.",
            picks=picks,
        )

    @staticmethod
    def _synthesise(traces: list[ToolCallTrace]) -> str:
        if not traces or not traces[-1].titles:
            return "I ran into trouble narrowing that down. Want to loosen one of the constraints?"
        lines = "\n".join(f"- **{t}**" for t in traces[-1].titles)
        return f"Here is where I landed:\n{lines}\n\nWant me to push further on any of these?"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

class AnthropicClient:
    provider = "anthropic"
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            }
        )

    def _post(self, messages: list[dict[str, Any]], system: str, temperature: float) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": 1400,
            "system": system,
            "messages": messages,
            "tools": T.anthropic_tools(),
            "tool_choice": {"type": "auto"},
            "temperature": temperature,
        }
        last_error = "unknown error"
        for attempt in range(3):
            try:
                resp = self.session.post(self.API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code == 200:
                return resp.json()

            body = resp.text[:400]
            if resp.status_code in (408, 429, 500, 502, 503, 529):
                last_error = f"HTTP {resp.status_code}: {body}"
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"Anthropic returned HTTP {resp.status_code}: {body}")

        raise LLMError(f"Anthropic request failed after 3 attempts ({last_error}).")

    # -- format translation -------------------------------------------------- #
    @staticmethod
    def _to_wire(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Internal format -> Anthropic wire format.

        Two wrinkles Anthropic is strict about:
          * assistant turns carrying tool calls become multi-block content
          * every `tool` message must be folded into a *single* following user
            turn as `tool_result` blocks
        """
        wire: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_results:
                wire.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }
                )
                continue

            flush()

            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for call in msg["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": _load_json_arg(call["function"].get("arguments")),
                        }
                    )
                wire.append({"role": "assistant", "content": blocks})
                continue

            wire.append({"role": role, "content": msg.get("content") or ""})

        flush()
        return wire

    # -- the loop ----------------------------------------------------------- #
    def respond(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> AgentReply:
        # Keep only conversational turns; Anthropic takes `system` out of band.
        convo = [m for m in (history or []) if m.get("role") in ("user", "assistant") and not m.get("tool_calls")]
        convo.append({"role": "user", "content": user_message})

        traces: list[ToolCallTrace] = []
        picks: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            data = self._post(self._to_wire(convo), SYSTEM_PROMPT, temperature)
            blocks = data.get("content", [])
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

            if not tool_uses:
                return AgentReply(
                    text=text or "I could not put together an answer. Could you rephrase that?",
                    tool_calls=traces,
                    provider=self.provider,
                    model=self.model,
                    picks=picks,
                )

            pairs = [(b["name"], b.get("input") or {}) for b in tool_uses]
            new_traces, outputs = run_tool_calls(pairs)
            traces.extend(new_traces)
            picks = _best_picks(outputs) or picks

            convo.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": b["id"], "function": {"name": b["name"], "arguments": _dump(b.get("input") or {})}}
                for b in tool_uses
            ]})
            convo.append(
                {
                    "role": "tool",
                    "content": _dump(outputs),
                    "tool_call_id": ",".join(b["id"] for b in tool_uses),
                }
            )

        return AgentReply(
            text=OpenAIClient._synthesise(traces),
            tool_calls=traces,
            provider=self.provider,
            model=self.model,
            note=f"Stopped after {MAX_TOOL_ITERATIONS} tool rounds.",
            picks=picks,
        )


# --------------------------------------------------------------------------- #
# Offline fallback: rule-based intent parsing over the same tools
# --------------------------------------------------------------------------- #

_NUM_WORDS = {
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "one twenty": 120, "one fifty": 150,
    "two": 2, "three": 3, "four": 4, "five": 5,
}

_TIME_RE = re.compile(
    r"(?P<op>under|below|less than|shorter than|within|at most|max|upto|up to|around|about)?\s*"
    r"(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|hrs?|h\b|minutes?|mins?|m\b)",
    re.I,
)
_HALF_HOUR = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?:and a half|½)\s*(?P<unit>hours?|hrs?)", re.I
)


@dataclass
class Intent:
    moods: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    max_minutes: int | None = None
    min_minutes: int | None = None
    kind: str | None = None
    free_text: str | None = None
    follow_up: bool = False


def parse_intent(message: str) -> Intent:
    """Best-effort NL -> structured constraints. No API key required."""
    text = C.clean(message)
    intent = Intent()

    # --- runtime ---------------------------------------------------------- #
    for match in _HALF_HOUR.finditer(message):
        hours = float(match.group("num")) + 0.5
        intent.max_minutes = int(hours * 60)
    for match in _TIME_RE.finditer(message):
        num = float(match.group("num"))
        unit = match.group("unit").lower()
        op = (match.group("op") or "").lower()
        minutes = int(num * 60) if unit.startswith(("h", "hr")) else int(num)
        if op in ("under", "below", "less than", "shorter than", "within", "at most", "max", "upto", "up to"):
            intent.max_minutes = minutes
        elif minutes >= 45:
            intent.max_minutes = minutes
        else:
            intent.min_minutes = minutes
    if re.search(r"\bshort(er|est)?\b|\bquick(er)?\b|\bbrief\b|\bcompact\b", text):
        intent.max_minutes = intent.max_minutes or 100
    if re.search(r"\blong(er)?\b|\bepic\b|\bslow but\b", text):
        intent.max_minutes = 999
    if not intent.max_minutes:
        for word, minutes in _NUM_WORDS.items():
            if re.search(rf"\b{word}[\s-]*(?:min|minute|minutes)\b", text):
                intent.max_minutes = minutes

    # --- language ---------------------------------------------------------- #
    for alias, canonical in C.LANGUAGE_ALIASES.items():
        if len(alias) < 3:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", text) and canonical not in intent.languages:
            intent.languages.append(canonical)

    # --- type -------------------------------------------------------------- #
    if re.search(r"\bshows?\b|\bseries\b|\bseasons?\b|\bepisodes?\b|\btv\b", text):
        intent.kind = "series"
    elif re.search(r"\bmovies?\b|\bfilms?\b|\bcinema\b", text):
        intent.kind = "movie"

    # --- mood then genre --------------------------------------------------- #
    # Canonical mood names first ("light", "cozy", "twisty"), then the alias
    # table ("lighthearted", "feel good", "date night", ...).
    for mood in sorted(C.MOOD_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(mood)}\b", text) and mood not in intent.moods:
            intent.moods.append(mood)
    for alias, canonical in C.MOOD_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text) and canonical not in intent.moods:
            intent.moods.append(canonical)
    for alias, canonical in C.GENRE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text) and canonical not in intent.genres:
            intent.genres.append(canonical)

    # A genre that is also a mood ("romantic comedy") should not double-count.
    intent.moods = [m for m in intent.moods if m not in intent.genres]

    # --- free text --------------------------------------------------------- #
    # Only when nothing structured was found. tokens() drops filler, so
    # "what about Aamir Khan movies?" yields ["aamir", "khan"] and not the
    # words "what" and "about", which match half the synopses in the catalog.
    if not (intent.moods or intent.genres or intent.max_minutes or intent.languages):
        content = C.tokens(message)
        if content:
            intent.free_text = " ".join(content)

    return intent


_ORDINALS = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "last": -1,
}


def _ordinal_pick(message: str, prior: list[str]) -> str | None:
    """"what is the second one about?" -> the second title from last reply."""
    if not prior:
        return None
    text = C.clean(message)
    for word, index in _ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            if index == -1:
                return prior[-1]
            if index < len(prior):
                return prior[index]
    return None


class HeuristicPlanner:
    """Runs the real tools from parsed constraints, then writes a friendly reply.

    Not a language model, and does not pretend to be one. It exists so that a
    fresh clone with no API key still demonstrates the tool-calling architecture
    end to end, and so the Flask layer is testable offline.
    """

    provider = "offline"
    model = "rule-based-intent-parser"

    def __init__(self, model: str = "rule-based-intent-parser") -> None:
        self.model = model

    def respond(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> AgentReply:
        intent = parse_intent(user_message)
        prior = _prior_titles(history or [])
        is_follow_up = bool(prior) and _looks_like_follow_up(user_message, intent)

        calls: list[tuple[str, dict[str, Any]]] = []
        prior_ids = None
        if is_follow_up and prior:
            resolved = C.resolve_titles(prior)
            prior_ids = [t["title"] for t in resolved] or prior

        # "what is the second one about?" - resolve the reference to one title
        # before doing anything else.
        singled = _ordinal_pick(user_message, prior or []) if is_follow_up else None
        if singled:
            detail = T.execute("get_title_details", {"title": singled})
            traces, outputs = run_tool_calls([("get_title_details", {"title": singled})])
            picked = detail.get("titles") or []
            if picked:
                t = picked[0]
                seasons = t.get("seasons")
                extra = f" It runs {seasons} season{'' if seasons == 1 else 's'}." if seasons else ""
                return AgentReply(
                    text=(
                        f"**{t['title']} ({t['year']})** - {t.get('full_overview') or t['hook']}\n\n"
                        f"{t['genres'] and ', '.join(t['genres'])} | {t['language']} | "
                        f"{t['runtime_minutes']} min{' per episode' if t.get('seasons') else ''} | "
                        f"rated {t['rating']}.{extra}\n\n"
                        "Want the full cast and director, or something similar to it?"
                    ),
                    tool_calls=traces,
                    provider=self.provider,
                    model=self.model,
                    used_fallback=True,
                    note="Resolved an ordinal reference from the previous turn.",
                    picks=picked,
                )

        axes = sum(
            bool(x)
            for x in (
                intent.moods,
                intent.genres,
                intent.languages,
                intent.max_minutes or intent.min_minutes,
            )
        )

        if intent.free_text:
            # A name, a person, a franchise: free-text search, then narrow the
            # hits by whatever structured filters were also present.
            calls.append(
                ("search_titles", {"query": intent.free_text, "limit": T.MAX_RESULTS})
            )
        elif axes >= 2:
            # Several constraints: one exact, single-pass search beats
            # intersecting several truncated result sets.
            args = {"limit": DEFAULT_POOL}
            if intent.moods:
                args["mood"] = intent.moods[0]
            if intent.genres:
                args["genre"] = intent.genres[0]
            if intent.languages:
                args["lang"] = intent.languages[0]
            if intent.max_minutes:
                args["max_minutes"] = intent.max_minutes
            if intent.min_minutes:
                args["min_minutes"] = intent.min_minutes
            if intent.kind and not prior_ids:
                args["type"] = intent.kind
            if prior_ids:
                args["restrict_to"] = prior_ids
            calls.append(("recommend_titles", args))
        else:
            # One constraint: use the dedicated tool, which the LLM path also prefers.
            args = {"limit": DEFAULT_POOL}
            if prior_ids:
                args["restrict_to"] = prior_ids
            if intent.kind:
                args["type"] = intent.kind
            if intent.moods:
                calls.append(("search_by_mood", {"mood": intent.moods[0], **args}))
            elif intent.genres:
                calls.append(("search_by_genre", {"genre": intent.genres[0], **args}))
            elif intent.languages:
                calls.append(("filter_by_language", {"lang": intent.languages[0], **args}))
            elif intent.max_minutes or intent.min_minutes:
                calls.append(
                    (
                        "filter_by_runtime",
                        {
                            "max_minutes": intent.max_minutes or 999,
                            **({} if not intent.min_minutes else {"min_minutes": intent.min_minutes}),
                            **args,
                        },
                    )
                )

        if not calls:
            total = C.catalog_facets()["total"]
            return AgentReply(
                text=(
                    "I did not catch a filter in that. I can search by **mood** "
                    "(light, cozy, mind-bending, emotional), **genre**, **runtime** "
                    "(under 90 minutes) or **language** (Hindi, Korean, Tamil...).\n\n"
                    "Try something like *something light, under 90 minutes, in Hindi*.\n\n"
                    f"The catalog currently holds {total} titles."
                ),
                tool_calls=[],
                provider=self.provider,
                model=self.model,
                used_fallback=True,
                note="No filter detected; showing capabilities.",
            )

        traces, outputs = run_tool_calls(calls)
        picks = _first_picks(outputs)
        if intent.free_text and (intent.moods or intent.languages or intent.max_minutes or intent.genres):
            picks = _narrow(picks, intent)
        text = _compose_reply(user_message, intent, picks, is_follow_up)
        return AgentReply(
            text=text,
            tool_calls=traces,
            provider=self.provider,
            model=self.model,
            used_fallback=True,
            note="No LLM key configured - running the tools through the offline planner.",
            picks=picks,
        )


def _prior_titles(history: list[dict[str, Any]]) -> list[str]:
    """Titles already put in front of the user, most recent assistant turn first."""
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        found = re.findall(r"\*\*(.+?)\*\*", msg.get("content") or "")
        names = [re.sub(r"\s*\(\d{4}\)\s*$", "", f).strip() for f in found]
        names = [n for n in names if 1 < len(n) < 70]
        if names:
            return names
    return []


def _looks_like_follow_up(message: str, intent: Intent) -> bool:
    """Is the user still talking about what was already recommended?

    Two signals, either is enough:
      * a referential phrase - "any of those", "more like the second", "shorter ones"
      * no new *hard* constraint. A runtime or mood mention narrows the previous
        picks ("shorter ones", "any of those with more comedy?"); a new language,
        genre or free-text query means a fresh search instead.
    """
    if re.search(
        r"\b(those|them|these|that|any of|of those|others|another|ones?|"
        r"first|second|third|last|one of|more|less|instead|actually|what about)\b",
        C.clean(message),
    ):
        return True
    return not (intent.moods or intent.genres or intent.languages or intent.free_text)


def _first_picks(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Titles to show, in order, from the tool outputs.

    The planner either issues one exact all-axes search or one single-axis
    search, so the first non-empty output is already the correct answer set.
    """
    for output in outputs:
        titles = output.get("titles") or []
        if titles:
            return titles
    return []


def _narrow(picks: list[dict[str, Any]], intent: Intent) -> list[dict[str, Any]]:
    """Apply structured constraints to free-text hits, keeping the tool's order."""
    if intent.languages:
        picks = [p for p in picks if p["language"] in intent.languages]
    if intent.max_minutes:
        picks = [p for p in picks if (p["runtime_minutes"] or 9999) <= intent.max_minutes]
    if intent.min_minutes:
        picks = [p for p in picks if (p["runtime_minutes"] or 0) >= intent.min_minutes]
    if intent.genres:
        wanted = set(intent.genres)
        picks = [p for p in picks if wanted & set(p["genres"])]
    if intent.moods:
        wanted_moods = set(intent.moods)
        picks = [p for p in picks if any(m in p["genres"] for m in wanted_moods)]
    return picks


def _compose_reply(
    message: str, intent: Intent, picks: list[dict[str, Any]], is_follow_up: bool
) -> str:
    if not picks:
        return (
            "Nothing in the catalog matched all of that at once. Loosen one filter "
            "and I will try again - for example drop the language, or let me go up to "
            "120 minutes."
        )

    bits: list[str] = []
    if intent.moods:
        bits.append(C.MOOD_MAP[intent.moods[0]]["label"])
    if intent.genres:
        bits.append(" and ".join(g.lower() for g in intent.genres))
    if intent.max_minutes and intent.max_minutes < 999:
        bits.append(f"under {intent.max_minutes} min")
    if intent.languages:
        bits.append("in " + " or ".join(intent.languages))
    described = ", ".join(bits) if bits else "what you asked for"

    shown = picks[:5]
    head = "Going back through those" if is_follow_up else "Here is what I found"
    lines = [f"{head}, {described}:"]
    for p in shown:
        runtime = f"{p['runtime_minutes']} min" if p["runtime_minutes"] else "runtime unknown"
        extra = f" · {p['seasons']} season{'' if p['seasons'] == 1 else 's'}" if p.get("seasons") else ""
        lines.append(
            f"- **{p['title']} ({p['year']})** — {p['hook']} "
            f"({runtime}{extra}, {p['rating']})"
        )
    lines.append("")
    lines.append("Want me to push any of these further, or try a different mood?")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def build_client() -> Any:
    """Pick a client from the environment, falling back to the offline planner."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

    if provider in ("anthropic", "claude") and anthropic_key:
        return AnthropicClient(anthropic_key, os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL))
    if provider in ("openai", "gpt") and openai_key:
        return OpenAIClient(openai_key, os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    if provider in ("anthropic", "claude"):
        raise LLMError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set.")
    if provider in ("openai", "gpt"):
        raise LLMError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")

    if openai_key and anthropic_key:
        return OpenAIClient(openai_key, os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    if openai_key:
        return OpenAIClient(openai_key, os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    if anthropic_key:
        return AnthropicClient(anthropic_key, os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL))

    return HeuristicPlanner()


def llm_status() -> dict[str, Any]:
    """What the UI shows in the header so the mode is never a mystery."""
    openai_key = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    anthropic_key = bool((os.getenv("ANTHROPIC_API_KEY") or "").strip())
    provider = (os.getenv("LLM_PROVIDER") or ("openai" if openai_key else "anthropic" if anthropic_key else "")).lower()
    client = build_client()
    return {
        "provider": client.provider,
        "model": client.model,
        "live": client.provider in ("openai", "anthropic"),
        "openai_key": openai_key,
        "anthropic_key": anthropic_key,
        "configured_provider": provider or None,
        "catalog_size": C.catalog_facets()["total"],
    }


def respond(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    client: Any | None = None,
) -> AgentReply:
    """The one function app.py needs. Degrades to the offline planner on failure."""
    client = client or build_client()
    try:
        return client.respond(user_message, history=history or [])
    except LLMError as exc:
        fallback = HeuristicPlanner()
        reply = fallback.respond(user_message, history=history or [])
        reply.note = f"LLM call failed ({exc}). Used the offline planner instead."
        reply.provider = "offline (llm error)"
        return reply
    except Exception as exc:  # noqa: BLE001 - never surface a traceback to the chat
        fallback = HeuristicPlanner()
        reply = fallback.respond(user_message, history=history or [])
        reply.note = f"Unexpected LLM error ({exc}). Used the offline planner instead."
        reply.provider = "offline (llm error)"
        return reply
