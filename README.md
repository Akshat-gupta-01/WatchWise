# WatchWise

A conversational movie and show discovery assistant for an OTT-style catalog.
You type how you feel, an LLM picks the right tools, the tools query the
dataset, and the LLM writes the reply.

```
you  ->  "something light, under 90 minutes, in Hindi"
LLM  ->  recommend_titles(mood="light", max_minutes=90, lang="Hindi")
tool ->  10 titles that satisfy all three constraints
LLM  ->  "Here are five that fit - Gullak, Little Things, ..."
```

---

## Folder structure

```
New folder/
├── app.py                    Flask app, routes, session memory store
├── llm.py                    Provider clients + the function-calling loop
├── tools.py                  Tool functions, JSON schemas, dispatcher
├── catalog.py                Data loading, normalisation, mood/scoring engine
├── fetch_tmdb.py             Optional: rebuild data/titles.json from TMDB
├── test_cineagent.py         51 tests, incl. a fake LLM server
├── requirements.txt
├── .env.example
├── data/
│   └── titles.json           263 films + series (committed, no API key needed)
├── static/
│   ├── app.js                chat client, markdown renderer, session id
│   └── style.css             chat shell, bubbles, title cards, tool trace
└── templates/
    └── index.html            Bootstrap chat UI
```

Layering, strictly one direction:

```
app.py  ->  llm.py  ->  tools.py  ->  catalog.py  ->  data/titles.json
```

`catalog.py` knows nothing about LLMs. `tools.py` knows nothing about HTTP.
`llm.py` never touches the dataset. That is what makes the whole thing testable
without an API key.

---

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
```

It runs immediately with no API key. To enable the real LLM:

```bash
# macOS / Linux
export OPENAI_API_KEY=sk-...

# PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."   # or this instead
```

Restart `app.py`. The badge in the header flips from **offline planner** to
**openai · gpt-4o-mini** when a key is picked up.

No vendor SDK is required - both providers are called over plain HTTP with
`requests`, so there is nothing extra to install or keep in sync.

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | – | enables the OpenAI client |
| `ANTHROPIC_API_KEY` | – | enables the Anthropic client |
| `LLM_PROVIDER` | auto | `openai` or `anthropic`, to disambiguate |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | |
| `TMDB_API_KEY` | – | only for `fetch_tmdb.py` |
| `CINEAGENT_PORT` | `5000` | |

---

## The tools

Declared once in `_TOOL_SPECS` in `tools.py`; the same entry produces the Python
function, the OpenAI schema, and the Anthropic schema.

### The four you asked for

| Tool | Arguments | Does |
|---|---|---|
| `search_by_genre(genre, limit, restrict_to, type)` | genre | best titles in a genre |
| `search_by_mood(mood, limit, restrict_to, type, min_rating)` | mood | matches a *feeling*, not a category |
| `filter_by_runtime(max_minutes, min_minutes, limit, restrict_to, type, sort_by)` | minutes | length ceiling; per-episode for series |
| `filter_by_language(lang, limit, restrict_to, type, min_rating)` | language | `Hindi`, `hi`, `bollywood`, `K-drama` all resolve |

### Supporting tools

| Tool | Does |
|---|---|
| `recommend_titles(mood, genre, lang, max_minutes, type, ...)` | all axes in **one pass** - use for 2+ constraints |
| `search_titles(query, limit, type)` | free text over title, cast, director, synopsis |
| `recommend_similar(title, limit)` | "what else like X?" |
| `get_title_details(title)` | full synopsis, director, cast, seasons |
| `list_catalog_facets()` | what actually exists, so the model never guesses a name |

### Why `recommend_titles` exists

Intersecting three truncated tool results is quietly wrong. Ask for
"light, under 90 minutes, in Hindi" and each single-axis tool returns its own
top 5; their intersection is often *empty* even though the catalog holds titles
that satisfy all three. So a composed tool takes every constraint at once and
returns only titles that satisfy all of them. Single-axis requests still use the
single-axis tools.

### Arguments are forgiving

The model says `bollywood`, `hi`, `sci fi`, `lighthearted`, `comfortable`,
`i want to cry` - all resolved through alias tables in `catalog.py`. When
something is reinterpreted, the tool says so in `notes` and the model can
mention it. An unknown genre returns the list of valid genres instead of an
empty result.

---

## How follow-ups work

Every search tool accepts `restrict_to`: a list of titles. That single parameter
is what makes conversation state useful.

```
turn 1  "something light, under 90 minutes, in Hindi"
        -> recommend_titles(mood="light", lang="Hindi", max_minutes=90)
        -> Gullak, Little Things, Bhaukaal, ...

turn 2  "any of those with more comedy?"
        -> recommend_titles(mood="funny", genre="Comedy",
                            restrict_to=["Gullak", "Little Things", ...])
        -> narrows to 5, ranks by comedy
```

The model is told to do this in the system prompt, and the tool trace in the UI
shows `restrict_to=[5]` so you can see it happened.

Server-side, `app.py` keeps the rolling history per session id (a
`threading.Lock`-guarded dict, bounded by session count, turn count and TTL).
`GET /api/tools` dumps every schema - useful for seeing exactly what the model
is allowed to call.

---

## The function-calling loop

In `llm.py`, one shape for both providers:

```
messages -> POST -> tool_calls?
                      yes -> execute in tools.py -> append results -> POST again
                      no  -> final text
capped at MAX_TOOL_ITERATIONS (5)
```

- **OpenAI** sends `tools=[{type: function, ...}]` with `tool_choice: "auto"`,
  appends `{role: "tool", tool_call_id, content}` per result.
- **Anthropic** sends `input_schema`, and `AnthropicClient._to_wire()` translates
  the internal message format: assistant turns become multi-block `tool_use`
  content, and tool results are folded into a single following user turn as
  `tool_result` blocks. Anthropic is strict about both, so that translation is
  unit-tested directly.
- Retries with exponential backoff on 429/5xx; a 400 surfaces as `LLMError`
  rather than a stack trace.

### No API key? It still works

`HeuristicPlanner` parses intent with rules and calls the **same tools**, so the
architecture is demonstrable and testable offline. It handles single constraints,
multi-constraint requests, follow-ups via `restrict_to`, and ordinal references
("the first one", "the second"). If a live LLM call fails mid-request,
`llm.respond()` falls back to it and says so in the response `note`.

---

## Dataset

263 hand-curated titles in `data/titles.json` - Indian cinema across 8
languages, plus Korean, Japanese, French, German, Spanish and English films and
series, with realistic runtimes, ratings, certification and vibe tags.

The `vibes` array is what makes mood search work. Each title carries 2-6 tags
from a controlled vocabulary (`cozy`, `mind-bending`, `bittersweet`, `gory`,
`stylish`, `nostalgic`...), and `MOOD_MAP` in `catalog.py` maps each mood to the
genres, vibes and keywords that satisfy it. `mood_score()` combines them and
subtracts for disqualifying tags, so "cozy" actively avoids horror.

Optional refresh:

```bash
export TMDB_API_KEY=...
python fetch_tmdb.py --pages 5     # overwrites data/titles.json
```

It infers `vibes` heuristically from genre, keywords and rating, so review the
output before trusting mood search on it. This product uses the TMDB API but is
not endorsed or certified by TMDB.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | chat UI |
| `GET` | `/api/status` | provider, model, catalog facets, starter prompts |
| `GET` | `/api/tools` | every tool with its JSON schema |
| `POST` | `/api/chat` | `{message, session_id}` -> `{reply, titles, tool_calls, ...}` |
| `POST` | `/api/reset` | `{session_id}` |
| `GET` | `/api/health` | liveness + catalog size |

`tool_calls` in the response carries name, arguments, summary and hit count -
the same data the UI renders as the collapsible trace.

---

## Tests

```bash
python test_cineagent.py      # 51 tests, ~1s
```

The valuable ones spin up a fake OpenAI/Anthropic HTTP server, so the
function-calling loop, the schema payloads, the tool-result feedback and the
Anthropic message translation are all verified without spending a token.
Also covered: every tool, alias resolution, `restrict_to` narrowing, the offline
planner, Flask routes and session isolation.

---

## Known limits

- Session memory is in-process. Restarting `app.py` clears it; multiple workers
  would each hold their own. Swap `SessionStore` for Redis if that matters.
- Titles are matched by name, not by TMDB ID, so `restrict_to` breaks on
  ambiguous names.
- Mood search is a keyword-scored heuristic over hand-written tags, not
  embeddings. It is predictable and inspectable, and it will occasionally
  surprise you.
- `data/titles.json` is a fixed snapshot; ratings do not update.
