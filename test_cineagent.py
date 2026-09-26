"""
test_cineagent.py
=================
Run with:  python test_cineagent.py     (no pytest needed)

Covers the parts that are easy to break and expensive to debug by hand:
  * every tool, including the argument-juggling and failure paths
  * the mood / genre / language / runtime alias resolution
  * the follow-up contract: `restrict_to` really narrows
  * the OpenAI and Anthropic function-calling loops, driven by a fake HTTP
    server, so the message-format translation is exercised without a key
  * Flask routes and session memory

The mocked-LLM tests are the important ones: they are the only way to verify
the tool-call loop without spending tokens.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import app as A
import catalog as C
import llm
import tools as T


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

class TestTools(unittest.TestCase):
    def test_every_tool_is_dispatchable(self):
        for name in T.TOOL_FUNCTIONS:
            out = T.execute(name, {} if name not in ("search_by_genre", "search_by_mood",
                                                    "filter_by_language",
                                                    "filter_by_runtime",
                                                    "search_titles", "get_title_details",
                                                    "recommend_similar") else None)
            self.assertIn("ok", out, name)

    def test_unknown_tool_fails_softly(self):
        out = T.execute("no_such_tool", {})
        self.assertFalse(out["ok"])
        self.assertEqual(out["count"], 0)

    def test_bad_arguments_fail_softly(self):
        out = T.execute("search_by_genre", {"nonexistent_arg": 1})
        self.assertFalse(out["ok"])
        self.assertTrue(out["notes"])

    def test_genre_aliases(self):
        for alias in ("comedy", "Comedy", "funny", "scary", "sci-fi", "psychological thriller"):
            out = T.execute("search_by_genre", {"genre": alias, "limit": 3})
            self.assertGreater(out["count"], 0, alias)

    def test_unknown_genre_returns_options(self):
        out = T.execute("search_by_genre", {"genre": "quantum musicals"})
        self.assertEqual(out["count"], 0)
        self.assertIn("Available genres", out["notes"][0])

    def test_language_aliases(self):
        for alias in ("Hindi", "hi", "hindi", "bollywood", "K-drama", "korean", "en"):
            out = T.execute("filter_by_language", {"lang": alias, "limit": 3})
            self.assertGreater(out["count"], 0, alias)

    def test_unknown_language_returns_options(self):
        out = T.execute("filter_by_language", {"lang": "Welsh"})
        self.assertEqual(out["count"], 0)
        self.assertIn("Available languages", out["notes"][0])

    def test_runtime_respects_ceiling(self):
        out = T.execute("filter_by_runtime", {"max_minutes": 90, "limit": 12})
        self.assertGreater(out["count"], 0)
        for t in out["titles"]:
            self.assertLessEqual(t["runtime_minutes"], 90)

    def test_runtime_shortest_sort(self):
        out = T.execute("filter_by_runtime", {"max_minutes": 200, "sort_by": "shortest", "limit": 6})
        runtimes = [t["runtime_minutes"] for t in out["titles"]]
        self.assertEqual(runtimes, sorted(runtimes))

    def test_runtime_impossible_window(self):
        out = T.execute("filter_by_runtime", {"max_minutes": 3})
        self.assertEqual(out["count"], 0)

    def test_mood_aliases(self):
        for alias in ("light", "lighthearted", "cozy", "feel-good", "feel good",
                      "mind-bending", "i want to cry", "weird but interesting", "nostalgic"):
            out = T.execute("search_by_mood", {"mood": alias, "limit": 3})
            self.assertGreater(out["count"], 0, alias)

    def test_unknown_mood_returns_options(self):
        out = T.execute("search_by_mood", {"mood": "vibey"})
        self.assertEqual(out["count"], 0)

    def test_recommend_titles_requires_all_constraints(self):
        out = T.execute(
            "recommend_titles",
            {"mood": "light", "lang": "Hindi", "max_minutes": 90, "limit": 12},
        )
        for t in out["titles"]:
            self.assertEqual(t["language"], "Hindi")
            self.assertLessEqual(t["runtime_minutes"], 90)

    def test_recommend_titles_impossible_combination(self):
        out = T.execute("recommend_titles", {"lang": "Hindi", "max_minutes": 2})
        self.assertEqual(out["count"], 0)

    def test_restrict_to_narrows_the_result_set(self):
        """The follow-up contract. This is the single most important test here."""
        wide = T.execute("search_by_genre", {"genre": "comedy", "limit": 12})
        picks = [t["title"] for t in wide["titles"]][:4]
        self.assertGreaterEqual(len(picks), 2)

        narrow = T.execute(
            "search_by_genre", {"genre": "comedy", "restrict_to": picks, "limit": 12}
        )
        self.assertLessEqual(narrow["count"], len(picks))
        for t in narrow["titles"]:
            self.assertIn(t["title"], picks)
        self.assertTrue(any("within" in n for n in narrow["notes"]))

    def test_restrict_to_unresolvable_falls_back(self):
        out = T.execute(
            "search_by_genre", {"genre": "comedy", "restrict_to": ["Not A Real Film"], "limit": 3}
        )
        self.assertGreater(out["count"], 0)
        self.assertTrue(any("full catalog" in n for n in out["notes"]))

    def test_type_filter(self):
        series = T.execute("search_by_genre", {"genre": "comedy", "type": "series", "limit": 6})
        for t in series["titles"]:
            self.assertEqual(t["type"], "series")

    def test_get_title_details(self):
        out = T.execute("get_title_details", {"title": "Andhadhun"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["titles"][0]["title"], "Andhadhun")
        self.assertIn("full_overview", out["titles"][0])

    def test_get_title_details_miss(self):
        out = T.execute("get_title_details", {"title": "Zzzz Not Real"})
        self.assertFalse(out["ok"])

    def test_recommend_similar_excludes_seed(self):
        out = T.execute("recommend_similar", {"title": "Inception", "limit": 5})
        self.assertNotIn("Inception", [t["title"] for t in out["titles"]])

    def test_free_text_search_ignores_filler_words(self):
        """"about" matches half the synopses; it must not drive results."""
        out = T.execute("search_titles", {"query": "what about Aamir Khan movies?", "limit": 6})
        titles = [t["title"] for t in out["titles"]]
        self.assertIn("3 Idiots", titles)
        self.assertNotIn("Oppenheimer", titles)

    def test_facets(self):
        out = T.execute("list_catalog_facets")
        self.assertIn("Comedy", out["genres"])
        self.assertIn("Hindi", out["languages"])


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

class TestCatalog(unittest.TestCase):
    def test_dataset_loads_and_is_normalised(self):
        titles = C.load_catalog()
        self.assertGreater(len(titles), 150)
        for t in titles:
            self.assertIn(t["type"], ("movie", "series"))
            self.assertTrue(t["genres"])
            self.assertTrue(t["language"])
            self.assertIsInstance(t["vibes"], list)

    def test_no_duplicate_titles(self):
        names = [t["title"] for t in C.load_catalog()]
        self.assertEqual(len(names), len(set(names)))

    def test_genre_names_are_canonical(self):
        """'Sci-Fi' and 'Science Fiction' must not both be in the facet list."""
        genres = set(C.catalog_facets()["genres"])
        self.assertIn("Science Fiction", genres)
        self.assertNotIn("Sci-Fi", genres)

    def test_title_matching_is_fuzzy(self):
        self.assertTrue(C.title_matches(C.load_catalog()[0], C.load_catalog()[0]["title"]))
        out = C.resolve_titles(["andhadhun"])
        self.assertEqual(out[0]["title"], "Andhadhun")


# --------------------------------------------------------------------------- #
# Intent parser + offline planner
# --------------------------------------------------------------------------- #

class TestOfflinePlanner(unittest.TestCase):
    def setUp(self):
        self.planner = llm.HeuristicPlanner()

    def test_intent_runtime_parsing(self):
        self.assertEqual(llm.parse_intent("under 90 minutes").max_minutes, 90)
        self.assertEqual(llm.parse_intent("less than 2 hours").max_minutes, 120)
        self.assertEqual(llm.parse_intent("something short").max_minutes, 100)
        self.assertEqual(llm.parse_intent("about 45 min").max_minutes, 45)

    def test_intent_language_and_mood(self):
        i = llm.parse_intent("something light, in Hindi")
        self.assertIn("Hindi", i.languages)
        self.assertIn("light", i.moods)

    def test_three_axis_request(self):
        r = self.planner.respond("something light, under 90 minutes, in Hindi")
        self.assertTrue(r.tool_calls)
        self.assertTrue(r.text)
        self.assertTrue(r.picks)

    def test_follow_up_uses_restrict_to(self):
        first = self.planner.respond("comedy movies")
        history = [
            {"role": "user", "content": "comedy movies"},
            {"role": "assistant", "content": first.text},
        ]
        second = self.planner.respond("any of those with more romance?", history=history)
        restricted = [c for c in second.tool_calls if c.arguments.get("restrict_to")]
        self.assertTrue(restricted, "follow-up must narrow via restrict_to")
        self.assertTrue(second.text.lower().startswith("going back through those"))

    def test_ordinal_follow_up_resolves_one_title(self):
        first = self.planner.respond("thriller movies")
        history = [
            {"role": "user", "content": "thriller movies"},
            {"role": "assistant", "content": first.text},
        ]
        r = self.planner.respond("tell me about the second one", history=history)
        self.assertEqual(r.tool_calls[0].name, "get_title_details")
        self.assertTrue(r.picks)

    def test_no_filter_returns_guidance(self):
        r = self.planner.respond("hello")
        self.assertTrue(r.text)
        self.assertEqual(r.tool_calls, [])

    def test_every_reply_has_text(self):
        for q in ["hi", "something scary", "a k-drama", "bollywood comedy under 2 hours",
                  "movies by Nolan", "mind-bending", "a comfortable watch"]:
            self.assertTrue(self.planner.respond(q).text.strip(), q)


# --------------------------------------------------------------------------- #
# LLM function-calling loop, against a fake provider
# --------------------------------------------------------------------------- #

class _FakeHandler(BaseHTTPRequestHandler):
    """Serves a scripted tool_use round, then a text answer."""

    script: list[dict] = []
    seen: list[dict] = []

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append(body)

        step = min(len(type(self).seen) - 1, len(type(self).script) - 1)
        payload = type(self).script[step]

        if self.path.endswith("/chat/completions"):
            if payload.get("_tool"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": payload["_tool"],
                            "arguments": json.dumps(payload.get("_args", {})),
                        },
                    }],
                }
            else:
                message = {"role": "assistant", "content": payload["_text"]}
            out = {"choices": [{"message": message, "finish_reason": "stop"}]}
        else:  # Anthropic messages
            if payload.get("_tool"):
                content = [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": payload["_tool"],
                    "input": payload.get("_args", {}),
                }]
            else:
                content = [{"type": "text", "text": payload["_text"]}]
            out = {
                "id": "msg_1",
                "role": "assistant",
                "model": "fake",
                "content": content,
                "stop_reason": "end_turn",
            }

        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class TestFunctionCallingLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _FakeHandler.script = []
        _FakeHandler.seen = []

    def test_openai_loop_calls_tool_then_answers(self):
        _FakeHandler.script = [
            {"_tool": "recommend_titles", "_args": {"mood": "light", "lang": "Hindi",
                                                    "max_minutes": 90}},
            {"_text": "Here are three: **A**, **B** and **C**."},
        ]
        client = llm.OpenAIClient("sk-test", "gpt-test", base_url=self.base)
        reply = client.respond("something light, under 90 minutes, in Hindi")

        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].name, "recommend_titles")
        self.assertTrue(reply.text.startswith("Here are three"))
        self.assertTrue(reply.picks)

        # the tool result must have been fed back as a `tool` role message
        second = _FakeHandler.seen[1]
        roles = [m["role"] for m in second["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        fed_back = json.loads(second["messages"][-1]["content"])
        self.assertTrue(fed_back["ok"])
        self.assertTrue(fed_back["titles"])

    def test_openai_sends_tool_schemas(self):
        _FakeHandler.script = [{"_text": "hi"}]
        llm.OpenAIClient("sk-test", "gpt-test", base_url=self.base).respond("hi")
        sent = _FakeHandler.seen[0]
        self.assertEqual(sent["tool_choice"], "auto")
        names = {t["function"]["name"] for t in sent["tools"]}
        self.assertLessEqual(set(T.CORE_TOOL_NAMES), names)

    def test_anthropic_loop_calls_tool_then_answers(self):
        _FakeHandler.script = [
            {"_tool": "search_by_mood", "_args": {"mood": "cozy"}},
            {"_text": "Try **Gullak** and **My Neighbor Totoro**."},
        ]
        client = llm.AnthropicClient("sk-ant-test", "claude-test")
        client.session.post = _patch(client.session)  # point at the fake server
        client._post = lambda messages, system, temperature: _anthropic_post(
            self.base, messages, system, temperature
        )

        reply = client.respond("something cozy")
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].name, "search_by_mood")
        self.assertIn("Gullak", reply.text)
        self.assertTrue(reply.picks)

    def test_anthropic_message_translation(self):
        """tool_calls become tool_use blocks; tool messages become tool_result."""
        wire = llm.AnthropicClient._to_wire([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "search_by_mood", "arguments": '{"mood":"cozy"}'}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ])
        self.assertEqual([m["role"] for m in wire], ["user", "assistant", "user", "assistant"])
        self.assertEqual(wire[1]["content"][0]["type"], "tool_use")
        self.assertEqual(wire[1]["content"][0]["input"], {"mood": "cozy"})
        self.assertEqual(wire[2]["content"][0]["type"], "tool_result")
        self.assertEqual(wire[2]["content"][0]["tool_use_id"], "t1")

    def test_openai_handles_stringified_arguments(self):
        self.assertEqual(llm._load_json_arg('{"a": 1}'), {"a": 1})
        self.assertEqual(llm._load_json_arg({"a": 1}), {"a": 1})
        self.assertEqual(llm._load_json_arg("not json"), {})
        self.assertEqual(llm._load_json_arg(None), {})

    def test_loop_is_capped(self):
        """A model that only ever calls tools must not loop forever."""
        _FakeHandler.script = [{"_tool": "list_catalog_facets", "_args": {}}]
        client = llm.OpenAIClient("sk-test", "gpt-test", base_url=self.base)
        reply = client.respond("loop please")
        self.assertLessEqual(len(reply.tool_calls), llm.MAX_TOOL_ITERATIONS)
        self.assertIn("Stopped after", reply.note)

    def test_llm_error_falls_back_to_offline_planner(self):
        class Broken:
            provider = "openai"
            model = "broken"

            def respond(self, *a, **k):
                raise llm.LLMError("HTTP 500: boom")

        reply = llm.respond("something light in Hindi", client=Broken())
        self.assertIn("offline", reply.provider)
        self.assertIn("LLM call failed", reply.note)
        self.assertTrue(reply.text)


def _patch(session):  # pragma: no cover - helper for the anthropic test
    return session.post


def _anthropic_post(base, messages, system, temperature):
    import requests
    return requests.post(
        f"{base}/v1/messages",
        json={"model": "claude-test", "messages": messages, "system": system},
        timeout=10,
    ).json()


# --------------------------------------------------------------------------- #
# Flask
# --------------------------------------------------------------------------- #

class TestFlask(unittest.TestCase):
    def setUp(self):
        A.app.config["TESTING"] = True
        self.client = A.app.test_client()

    def test_index_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(b"WatchWise" in res.data or b"CineAgent" in res.data)

    def test_health(self):
        data = self.client.get("/api/health").get_json()
        self.assertTrue(data["ok"])
        self.assertGreater(data["catalog_titles"], 150)

    def test_status(self):
        data = self.client.get("/api/status").get_json()
        self.assertIn("tools", data)
        self.assertTrue(data["starter_prompts"])
        self.assertIn("live", data)

    def test_tools_endpoint_exposes_schemas(self):
        tools = self.client.get("/api/tools").get_json()
        names = {t["name"] for t in tools}
        self.assertLessEqual(set(T.CORE_TOOL_NAMES), names)
        for t in tools:
            self.assertEqual(t["parameters"]["type"], "object")

    def test_chat_rejects_empty_message(self):
        self.assertEqual(self.client.post("/api/chat", json={"message": "  "}).status_code, 400)

    def test_chat_rejects_overlong_message(self):
        res = self.client.post("/api/chat", json={"message": "x" * 5000})
        self.assertEqual(res.status_code, 400)

    def test_chat_returns_reply_and_trace(self):
        data = self.client.post("/api/chat", json={"message": "something light in Hindi"}).get_json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["reply"])
        self.assertTrue(data["tool_calls"])
        self.assertIn("session_id", data)

    def test_session_memory_keeps_context(self):
        self.client.post("/api/chat", json={"message": "comedy movies", "session_id": "test-sess"})
        second = self.client.post(
            "/api/chat",
            json={"message": "any of those with more romance?", "session_id": "test-sess"},
        ).get_json()

        self.assertEqual(second["turn"], 2)
        restricted = [c for c in second["tool_calls"] if c["arguments"].get("restrict_to")]
        self.assertTrue(restricted, "second turn must reuse the first turn's context")
        self.assertIn("going back through", second["reply"].lower())

    def test_sessions_are_isolated(self):
        a = self.client.post("/api/chat", json={"message": "comedy", "session_id": "A"}).get_json()
        b = self.client.post("/api/chat", json={"message": "any of those?", "session_id": "B"}).get_json()
        self.assertEqual(b["turn"], 1)
        self.assertNotEqual(a["session_id"], "B")

    def test_reset_clears_memory(self):
        self.client.post("/api/chat", json={"message": "comedy", "session_id": "R"})
        self.client.post("/api/reset", json={"session_id": "R"})
        after = self.client.post("/api/chat", json={"message": "hi", "session_id": "R"}).get_json()
        self.assertEqual(after["turn"], 1)

    def test_history_is_trimmed(self):
        session = A.ChatSession(id="t")
        for i in range(A.MAX_TURNS_KEPT + 8):
            session.remember(f"q{i}", f"a{i}", [])
        self.assertLessEqual(len(session.history), A.MAX_TURNS_KEPT * 2)
        self.assertEqual(session.history[0]["role"], "user")


if __name__ == "__main__":
    unittest.main(verbosity=2)
