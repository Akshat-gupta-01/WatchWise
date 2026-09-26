"""
app.py
======
Flask front end and the session-memory layer.

Routes
------
    GET  /                the chat UI
    GET  /api/status      which LLM is wired up, catalog size, starter prompts
    GET  /api/tools       every tool the assistant can call, with its schema
    POST /api/chat        one turn: {message, session_id} -> reply + tool traces
    POST /api/reset       wipe a session
    GET  /api/health      liveness + catalog sanity check

Session memory lives in a small in-process store keyed by an opaque session id
the browser keeps in localStorage. Each session holds the rolling message
history handed to the LLM, so "any of those with more comedy?" has something to
refer to. The store is bounded (max sessions, max turns, TTL) and locked, since
Flask's dev server is threaded.

Run it with:  python app.py     (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from flask import Flask, jsonify, render_template, request

import catalog as C
import llm
import tools as T

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HOST = os.getenv("CINEAGENT_HOST", "127.0.0.1")
PORT = int(os.getenv("CINEAGENT_PORT", "5000"))
DEBUG = os.getenv("CINEAGENT_DEBUG", "1") not in ("0", "false", "False")

MAX_MESSAGE_CHARS = 1000
MAX_SESSIONS = 500
SESSION_TTL_SECONDS = 60 * 60 * 3  # 3 hours
MAX_TURNS_KEPT = 12  # user+assistant pairs retained per session

STARTER_PROMPTS = [
    "something light, under 90 minutes, in Hindi",
    "a mind-bending thriller I haven't seen",
    "cozy show for a rainy evening",
    "what's like Andhadhun?",
    "a K-drama in under an hour",
]

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024


# --------------------------------------------------------------------------- #
# Session memory
# --------------------------------------------------------------------------- #

@dataclass
class ChatSession:
    id: str
    history: list[dict[str, str]] = field(default_factory=list)
    titles_shown: list[str] = field(default_factory=list)
    turns: int = 0
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def remember(self, user_text: str, reply_text: str, titles: list[str]) -> None:
        """Append the exchange and trim to the context window."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply_text})
        for name in titles:
            if name not in self.titles_shown:
                self.titles_shown.append(name)
        self.titles_shown = self.titles_shown[-40:]

        # Keep the history to the most recent MAX_TURNS_KEPT exchanges, and never
        # start on a dangling assistant message.
        if len(self.history) > MAX_TURNS_KEPT * 2:
            self.history = self.history[-MAX_TURNS_KEPT * 2 :]
        if self.history and self.history[0]["role"] != "user":
            self.history = self.history[1:]

        self.turns += 1
        self.last_seen = time.time()

    def llm_history(self) -> list[dict[str, str]]:
        """Plain user/assistant turns only - no tool plumbing."""
        return [m for m in self.history if m.get("role") in ("user", "assistant")]


class SessionStore:
    """Bounded, thread-safe, in-process session store."""

    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl: int = SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        self._max = max_sessions
        self._ttl = ttl

    def get(self, session_id: str | None) -> ChatSession:
        self._evict_expired()
        if session_id:
            with self._lock:
                existing = self._sessions.get(session_id)
                if existing:
                    existing.last_seen = time.time()
                    return existing
        fresh = ChatSession(id=session_id or uuid.uuid4().hex)
        with self._lock:
            self._sessions[fresh.id] = fresh
            self._evict_overflow()
        return fresh

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            for key in [k for k, v in self._sessions.items() if v.last_seen < cutoff]:
                self._sessions.pop(key, None)

    def _evict_overflow(self) -> None:
        if len(self._sessions) <= self._max:
            return
        for key, _ in sorted(self._sessions.items(), key=lambda kv: kv[1].last_seen)[
            : len(self._sessions) - self._max
        ]:
            self._sessions.pop(key, None)


SESSIONS = SessionStore()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "catalog_titles": len(C.load_catalog()),
            "active_sessions": SESSIONS.count(),
        }
    )


@app.get("/api/status")
def status() -> Any:
    info = llm.llm_status()
    facets = C.catalog_facets()
    return jsonify(
        {
            **info,
            "starter_prompts": STARTER_PROMPTS,
            "tools": list(T.TOOL_FUNCTIONS),
            "catalog": {
                "titles": facets["total"],
                "genres": facets["genres"][:14],
                "languages": facets["languages"],
                "moods": facets["moods"],
                "runtime_range": facets["runtime_range"],
            },
        }
    )


@app.get("/api/tools")
def list_tools() -> Any:
    return jsonify(
        [
            {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            }
            for spec in T.REGISTRY.values()
        ]
    )


@app.post("/api/chat")
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()

    if not message:
        return jsonify({"ok": False, "error": "Message is empty."}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Message too long (max {MAX_MESSAGE_CHARS} characters).",
                }
            ),
            400,
        )

    session = SESSIONS.get(payload.get("session_id"))

    reply = llm.respond(message, history=session.llm_history())
    session.remember(message, reply.text, [t["title"] for t in reply.picks])

    return jsonify(
        {
            "ok": True,
            "session_id": session.id,
            "reply": reply.text,
            "titles": reply.picks,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "summary": call.summary,
                    "count": call.count,
                    "titles": call.titles,
                }
                for call in reply.tool_calls
            ],
            "provider": reply.provider,
            "model": reply.model,
            "used_fallback": reply.used_fallback,
            "note": reply.note,
            "turn": session.turns,
        }
    )


@app.post("/api/reset")
def reset() -> Any:
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if session_id:
        SESSIONS.reset(str(session_id))
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main() -> None:
    info = llm.llm_status()
    facets = C.catalog_facets()

    print("=" * 66)
    print("  WatchWise (developed by Akshat Gupta and Shekhar Rana)")
    print("=" * 66)
    print(f"  Catalog   : {facets['total']} titles from data/titles.json")
    print(f"  LLM       : {info['provider']} ({info['model']})")
    if info["live"]:
        print("              function calling enabled against the live API")
    else:
        print("              no API key found - running the offline intent planner")
        print("              set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable the LLM")
    print(f"  Tools     : {', '.join(T.TOOL_FUNCTIONS)}")
    print(f"  Serving   : http://{HOST}:{PORT}")
    print("=" * 66)

    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)


if __name__ == "__main__":
    main()
