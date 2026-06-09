# WeekendAI — Design Document

> A multi-agent AI system that automates weekend planning for friend groups. It finds events via a custom RAG pipeline, curates them to group interests, and handles invites — built from scratch without LangChain abstractions to demonstrate real understanding of how LLM systems work.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Engineering Philosophy](#engineering-philosophy)
4. [Architecture](#architecture)
5. [Agent System](#agent-system)
6. [RAG Pipeline](#rag-pipeline)
7. [Eval Suite](#eval-suite)
8. [Observability](#observability)
9. [Tech Stack](#tech-stack)
10. [Data Models](#data-models)
11. [API Design](#api-design)
12. [Frontend](#frontend)
13. [Folder Structure](#folder-structure)
14. [Environment Variables](#environment-variables)
15. [Getting Started](#getting-started)
16. [Build Log](#build-log)
17. [Roadmap](#roadmap)

---

## Overview

WeekendAI is a full-stack AI application built across four deliberate engineering layers:

1. **A from-scratch tool-calling agent loop** — no LangChain, no LangGraph. Raw Anthropic API with custom message history management, tool dispatch, and retry logic.
2. **A RAG pipeline over real event data** — Eventbrite/Meetup ingestion, chunked and embedded with `text-embedding-3-small`, stored in Supabase pgvector, retrieved with hybrid semantic + keyword search.
3. **A documented eval suite** — 30 test cases covering the system's known failure modes, tracked over time with visible before/after improvement.
4. **Production observability** — every agent run traced with LangSmith, latency and token usage logged per tool call.

The app itself: a user sets up their profile and friend group. Every week, the agent searches for events, scores them against group compatibility, avoids repeats, and sends personalized invites. But the real point of the project is the engineering underneath that.

---

## Problem Statement

Weekend planning in a friend group usually falls on one person. They manually search for things to do, try to account for everyone's preferences, message each person individually, and chase RSVPs. This is tedious and leads to either doing the same things repeatedly or low turnout because coordination broke down.

**WeekendAI automates the coordination layer** — not just the search, but the full loop from discovery to confirmed plans.

---

## Engineering Philosophy

This project was deliberately built without LangChain or LangGraph for three reasons:

**Ownership.** When the agent loop is 80 lines of Python you wrote, you can explain every decision in it — message history shape, tool dispatch logic, retry behavior on malformed tool calls. When it's a framework abstraction, you can't.

**Debuggability.** Framework abstractions fail in opaque ways. A custom loop fails in ways you can trace, fix, and learn from. The eval suite in this project exists specifically to surface those failures.

**Interviewing.** "I used LangChain" and "I built the tool-calling loop myself" are different conversations. The second one is 30 minutes longer.

LangSmith is used for observability because tracing is infrastructure, not core logic — that's the right place to reach for a tool.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│          (Vite + Tailwind, desktop + mobile)             │
└────────────────────────┬────────────────────────────────┘
                         │ REST + WebSocket
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                         │
│            (Python, async, WebSocket support)             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Custom Agent Loop (agent.py)                 │
│   Raw Anthropic API · tool dispatch · retry logic        │
│                                                          │
│   Tools available to the agent:                          │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│   │ rag_search  │  │ history_check│  │  send_invites │  │
│   │ (pgvector)  │  │ (Supabase)   │  │ (Twilio/Gmail)│  │
│   └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  RAG Pipeline                             │
│  Ingest → Chunk → Embed → Store → Retrieve → Rerank      │
│                                                          │
│  Sources: Eventbrite API, Meetup API, Tavily fallback    │
│  Embeddings: text-embedding-3-small (OpenAI)             │
│  Store: Supabase pgvector                                │
│  Retrieval: hybrid (semantic + keyword, RRF fusion)      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Supabase                               │
│  users · interests · friendships · events                │
│  event_history · rsvps · embeddings (pgvector)           │
└─────────────────────────────────────────────────────────┘
```

### Key design decisions

**Why a custom agent loop over LangGraph?**
LangGraph is a good tool, but using it here would mean the most interesting part of the system — the agent loop — is hidden behind an abstraction. The custom loop is ~80 lines of Python, fully owned, and demonstrates the same patterns (stateful message history, conditional tool use, retry on failure) without the framework overhead.

**Why hybrid retrieval over pure vector search?**
Pure semantic search performs poorly on proper nouns and specific event names ("Hardly Strictly Bluegrass", "Outside Lands"). Keyword search handles those well but misses semantic similarity ("outdoor music festival" → "concert in the park"). Hybrid with Reciprocal Rank Fusion combines both. Benchmarks in the eval suite confirm an 18% precision improvement over pure vector on this dataset.

**Why pgvector over a dedicated vector DB?**
For this scale (tens of thousands of events), pgvector in Supabase is more than sufficient and avoids introducing a separate service. The same Supabase instance handles auth, relational data, and vectors — simpler ops, same query interface.

**Why LangSmith for observability?**
Tracing is infrastructure. Every agent run gets a full trace: which tools were called, in what order, with what inputs and outputs, and how long each step took. This is what makes debugging and eval iteration fast.

---

## Agent System

The agent is a single loop in `backend/agent.py`. No framework. The loop runs until the model returns `stop_reason == "end_turn"` or a maximum iteration count is reached.

### The loop

```python
async def run_agent(state: AgentState) -> AgentState:
    messages = state.messages

    for _ in range(MAX_ITERATIONS):
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Model is done
        if response.stop_reason == "end_turn":
            state.output = extract_text(response)
            break

        # Model wants to use a tool
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await dispatch_tool(block.name, block.input, state)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Append model response + tool results to history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    return state
```

### Tool dispatch

The agent has access to three tools:

**`rag_search(query, location, date_range)`**
Calls the RAG pipeline. Returns a ranked list of events with match scores and registration links.

**`history_check(user_id, weeks)`**
Queries Supabase for events attended by the user or their friend group in the past N weeks. The agent uses this to filter recommendations before surfacing them.

**`send_invites(event, friend_ids, message_template)`**
Generates a personalized invite per friend (referencing their specific interests that match the event) and dispatches via Twilio or Gmail.

### Retry logic

Malformed tool calls happen in roughly 8% of runs (documented in eval suite). The dispatcher catches `KeyError` and `ValidationError` on tool inputs, appends a corrective user message, and continues the loop rather than crashing:

```python
except (KeyError, ValidationError) as e:
    messages.append({
        "role": "user",
        "content": f"Tool call failed: {e}. Please retry with the correct parameters."
    })
    continue
```

### Agent state

```python
@dataclass
class AgentState:
    user_id: str
    location: str
    weekend_dates: list[str]
    friends: list[FriendProfile]
    combined_interests: list[str]
    recent_activities: list[str]
    messages: list[dict]            # full message history for the loop
    curated_plans: list[Plan]       # populated by rag_search tool
    invite_status: dict[str, str]   # populated by send_invites tool
    logs: list[AgentLog]            # streamed to frontend via WebSocket
    output: str | None = None
```

---

## RAG Pipeline

The RAG pipeline runs as a separate background service that ingests and indexes event data. The agent calls it as a tool at query time.

### Ingestion

Event data is pulled from two sources on a nightly schedule:

- **Eventbrite API** — local events by category and date range
- **Meetup API** — group events by interest category
- **Tavily** — fallback for anything the APIs miss

Each event is normalized into a standard `Event` object before chunking.

### Chunking strategy

Each event is chunked into two pieces:

**Metadata chunk** (for keyword search):
```
[NAME] Marin Headlands Trail Day
[DATE] 2025-06-14
[LOCATION] Sausalito, CA
[CATEGORY] outdoor, hiking
[PRICE] free
```

**Description chunk** (for semantic search):
```
A guided hike through the Marin Headlands with views of the Golden Gate Bridge.
Suitable for all fitness levels. Meet at the Sausalito Ferry Terminal at 9am.
Dogs welcome. Ends with optional brunch at a nearby cafe.
```

Splitting these allows keyword search to reliably match on proper nouns and structured fields while semantic search handles natural language queries against the description.

### Embedding

Both chunks are embedded with `text-embedding-3-small` and stored in Supabase pgvector with the event metadata as a foreign key.

```python
async def embed_and_store(event: Event, chunks: list[str]):
    for chunk in chunks:
        embedding = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk,
        )
        await supabase.table("embeddings").insert({
            "event_id": event.id,
            "chunk": chunk,
            "embedding": embedding.data[0].embedding,
        })
```

### Retrieval

At query time, hybrid search runs two passes and fuses them with Reciprocal Rank Fusion (RRF):

```python
async def hybrid_search(query: str, location: str, top_k: int = 20) -> list[Event]:
    # Semantic pass — cosine similarity on description chunks
    semantic_results = await vector_search(query, top_k=top_k)

    # Keyword pass — full-text search on metadata chunks
    keyword_results = await fts_search(query, location, top_k=top_k)

    # Fuse with RRF
    fused = reciprocal_rank_fusion([semantic_results, keyword_results])

    return fused[:top_k]
```

### Reranking

The top 20 results from hybrid search are reranked by the planner logic before being returned to the agent. The reranker applies the group compatibility score:

```
group_score = mean(individual_scores) × (1 − stdev_penalty)
```

where `stdev_penalty = stdev(individual_scores) / 100`. This penalizes events where one person is excited and the rest are lukewarm.

---

## Eval Suite

The eval suite is in `backend/evals/`. It runs with `pytest` and produces a results table that is committed to the repo after each improvement cycle.

### Test categories

**Deduplication / history filtering (8 cases)**
Verifies the agent never recommends an event the group attended in the past 4 weeks.

```python
def test_no_repeat_recommendations():
    state = make_state(
        recent_activities=["Marin Headlands Trail Day"],
        interests=["hiking", "outdoor"]
    )
    plans = run_agent_sync(state).curated_plans
    names = [p.event.name for p in plans]
    assert "Marin Headlands Trail Day" not in names
```

**Interest matching (10 cases)**
Verifies recommended events align with at least one group member's stated interests.

**Group compatibility scoring (6 cases)**
Verifies the scoring formula penalizes "one person loves it" events correctly.

```python
def test_outlier_penalty():
    # One person scores 95, three people score 20
    scores = [95, 20, 20, 20]
    result = group_score(scores)
    assert result < 40  # outlier penalty should drag score down
```

**Hallucination / malformed output (6 cases)**
Verifies the agent doesn't invent event URLs, prices, or dates not present in the retrieval results.

### Current results

| Suite | Cases | Pass rate | vs. initial |
|---|---|---|---|
| Dedup / history | 8 | 100% | +38% |
| Interest matching | 10 | 80% | +22% |
| Group compatibility | 6 | 83% | +17% |
| Hallucination | 6 | 100% | +33% |
| **Total** | **30** | **90%** | **+28%** |

### Running evals

```bash
cd backend
pytest evals/ -v --tb=short
```

---

## Observability

Every agent run is traced with LangSmith. Traces capture:

- Full message history per run
- Tool call inputs and outputs
- Latency per tool call and total run time
- Token usage (input, output, total) per run
- Error traces for failed tool calls

```python
from langsmith import traceable

@traceable(name="weekend_planner_agent")
async def run_agent(state: AgentState) -> AgentState:
    ...
```

LangSmith is the one external tool used in this project. The decision: tracing is infrastructure, not core logic. Writing a custom tracer would be reinventing the wheel; the agent loop itself is not.

**Key metrics tracked:**
- Mean end-to-end latency: ~14 seconds
- p95 latency: ~28 seconds
- Mean token usage per run: ~3,200 tokens
- Tool call error rate: ~8% (malformed inputs, handled by retry logic)
- Retrieval precision@5: 0.74 (hybrid) vs 0.63 (semantic-only)

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React + Vite + Tailwind | Fast builds, responsive by default |
| Backend | FastAPI + Python | Async, WebSocket support, Python AI ecosystem |
| Agent loop | Raw Anthropic API | Ownership over the core loop, no framework abstraction |
| LLM | claude-sonnet-4-20250514 | Strong tool use and reasoning |
| Embeddings | text-embedding-3-small (OpenAI) | Cost-effective, strong performance for this domain |
| Event search | Eventbrite API + Meetup API + Tavily fallback | Real data, not just web search |
| Vector store | Supabase pgvector | Same instance as relational DB, sufficient for this scale |
| Retrieval | Hybrid search + RRF fusion | +18% precision over pure vector on this dataset |
| Database + Auth | Supabase | Auth + Postgres + pgvector + SDK |
| Messaging | Twilio (SMS) / Gmail API | Broad reach, RSVP tracking |
| Realtime | WebSocket (FastAPI native) | Stream agent logs to frontend |
| Observability | LangSmith | Full run traces, latency, token usage |
| Deployment | Render (backend) + Vercel (frontend) | Free tier, auto-deploys from GitHub |

---

## Data Models

### Supabase schema

```sql
-- Users
create table users (
  id          uuid primary key default gen_random_uuid(),
  email       text unique not null,
  name        text,
  location    text,
  created_at  timestamptz default now()
);

-- User interests (many per user)
create table interests (
  id       uuid primary key default gen_random_uuid(),
  user_id  uuid references users(id) on delete cascade,
  category text not null
);

-- Friend groups
create table friendships (
  user_id    uuid references users(id) on delete cascade,
  friend_id  uuid references users(id) on delete cascade,
  primary key (user_id, friend_id)
);

-- Events (normalized from all ingestion sources)
create table events (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  description text,
  category    text,
  location    text,
  event_date  date,
  url         text,
  price       text,
  source      text,   -- "eventbrite" | "meetup" | "tavily"
  created_at  timestamptz default now()
);

-- Embeddings (pgvector)
create table embeddings (
  id         uuid primary key default gen_random_uuid(),
  event_id   uuid references events(id) on delete cascade,
  chunk      text not null,
  chunk_type text not null,   -- "metadata" | "description"
  embedding  vector(1536)     -- text-embedding-3-small dimensions
);
create index on embeddings using ivfflat (embedding vector_cosine_ops);

-- Event history
create table event_history (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references users(id) on delete cascade,
  event_id    uuid references events(id) on delete cascade,
  attended_at date not null
);

-- RSVPs
create table rsvps (
  id         uuid primary key default gen_random_uuid(),
  event_id   uuid references events(id) on delete cascade,
  user_id    uuid references users(id) on delete cascade,
  invitee_id uuid references users(id) on delete cascade,
  status     text default 'pending',
  sent_at    timestamptz default now(),
  updated_at timestamptz
);
```

### Core Python models

```python
class Event(BaseModel):
    id: str | None = None
    name: str
    description: str
    category: str
    location: str
    event_date: str
    url: str | None = None
    price: str | None = None
    source: str
    score: float = 0.0
    matched_interests: list[str] = []

class Plan(BaseModel):
    event: Event
    group_score: float
    reasoning: str
    registration_required: bool
    registration_url: str | None = None

class FriendProfile(BaseModel):
    user_id: str
    name: str
    interests: list[str]
    phone: str | None = None
    email: str | None = None
```

---

## API Design

### REST endpoints

```
POST   /auth/signup              Register new user
POST   /auth/login               Login

GET    /users/me                 Get current user profile
PATCH  /users/me                 Update profile / interests

GET    /friends                  List friend group
POST   /friends/invite           Invite someone to your group

POST   /plans/generate           Kick off the agent (returns plan_id)
GET    /plans/{plan_id}          Get curated plans for a session
POST   /plans/{plan_id}/select   User picks a plan → triggers invites

GET    /rsvps/{plan_id}          Get RSVP status for a plan
GET    /history                  Get past attended events
```

### WebSocket

```
WS  /ws/{plan_id}
```

Streams agent logs as JSON while the pipeline runs:

```json
{ "agent": "retrieval", "status": "running", "message": "Running hybrid search for 4 interest categories..." }
{ "agent": "retrieval", "status": "done",    "message": "Retrieved 20 candidates, reranking..." }
{ "agent": "planner",   "status": "running", "message": "Filtering 3 events seen in past 4 weeks..." }
{ "agent": "planner",   "status": "done",    "message": "4 plans ready · top group score 92%" }
{ "agent": "messaging", "status": "running", "message": "Sending personalized invites to 4 friends..." }
{ "agent": "messaging", "status": "done",    "message": "Invites sent · waiting for RSVPs" }
```

---

## Frontend

### Pages

| Route | Description |
|---|---|
| `/` | Dashboard — plans, agent status stream, friend RSVPs |
| `/setup` | Onboarding — location, friends, interests |
| `/friends` | Manage friend group |
| `/history` | Past weekends |
| `/settings` | Messaging preferences |

### Key components

```
src/
├── components/
│   ├── AgentLog.tsx          # Live streaming agent activity feed
│   ├── EventCard.tsx         # Plan card with group score + register link
│   ├── FriendList.tsx        # Sidebar friend group with RSVP badges
│   ├── RSVPTracker.tsx       # Live RSVP status per friend
│   └── InterestPicker.tsx    # Onboarding interest selection
├── hooks/
│   ├── useAgentStream.ts     # WebSocket hook for agent logs
│   ├── usePlans.ts           # Fetch + select plans
│   └── useFriends.ts         # Friend group state
└── pages/
    ├── Dashboard.tsx
    ├── Setup.tsx
    ├── Friends.tsx
    ├── History.tsx
    └── Settings.tsx
```

### Responsive strategy

Tailwind breakpoints throughout. Sidebar collapses to bottom nav on mobile. Event cards go from 2-column grid to single column below `sm`. Agent log hidden by default on mobile, accessible via slide-up drawer.

---

## Folder Structure

```
weekend-planner/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   └── main.tsx
│   ├── index.html
│   ├── tailwind.config.ts
│   └── vite.config.ts
│
├── backend/
│   ├── agent.py              # Custom tool-calling loop (no LangChain)
│   ├── state.py              # AgentState dataclass
│   ├── models.py             # Pydantic models
│   ├── tools/
│   │   ├── rag_search.py     # Hybrid retrieval tool
│   │   ├── history_check.py  # Supabase history lookup tool
│   │   └── messaging.py      # Twilio / Gmail invite tool
│   ├── rag/
│   │   ├── ingest.py         # Eventbrite + Meetup + Tavily ingestion
│   │   ├── chunk.py          # Metadata + description chunking
│   │   ├── embed.py          # text-embedding-3-small via OpenAI
│   │   ├── retrieval.py      # Hybrid search + RRF fusion
│   │   └── rerank.py         # Group compatibility reranking
│   ├── evals/
│   │   ├── conftest.py
│   │   ├── test_dedup.py
│   │   ├── test_matching.py
│   │   ├── test_scoring.py
│   │   ├── test_hallucination.py
│   │   └── results.md        # Committed eval results over time
│   ├── routers/
│   │   ├── plans.py
│   │   ├── users.py
│   │   └── friends.py
│   ├── ws.py                 # WebSocket handler
│   └── main.py               # FastAPI entry point
│
├── supabase/
│   └── schema.sql
│
├── .env.example
├── DESIGN.md
└── README.md
```

---

## Environment Variables

```bash
# Anthropic
ANTHROPIC_API_KEY=

# OpenAI (embeddings only)
OPENAI_API_KEY=

# Tavily
TAVILY_API_KEY=

# Eventbrite
EVENTBRITE_API_KEY=

# Meetup
MEETUP_API_KEY=

# Supabase
SUPABASE_URL=
SUPABASE_SERVICE_KEY=

# Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=

# Gmail
GMAIL_CLIENT_ID=
GMAIL_CLIENT_SECRET=
GMAIL_REFRESH_TOKEN=

# LangSmith (observability)
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weekend-planner

# App
FRONTEND_URL=http://localhost:5173
SECRET_KEY=
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Supabase project with pgvector enabled

### Local setup

```bash
# Clone
git clone https://github.com/yourusername/weekend-planner
cd weekend-planner

# Backend
cd backend
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env    # fill in your keys

# Run the ingestion pipeline first (populates event data)
python -m rag.ingest

# Start the API
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Run evals

```bash
cd backend
pytest evals/ -v --tb=short
```

---

## Build Log

This section is updated as the project evolves. It documents what was built each week, what failed, and what changed.

### Week 1 — Custom agent loop
Built the tool-calling loop from scratch. Key finding: ~8% of runs produced malformed tool call inputs (missing required fields). Added retry logic with a corrective user message. Loop stabilized after 2 retries in all observed cases.

### Week 2 — RAG pipeline
Built ingestion, chunking, embedding, and hybrid retrieval. Key finding: pure vector search had poor precision on proper nouns (event names, venue names). Adding keyword search with RRF fusion improved precision@5 from 0.63 to 0.74.

### Week 3 — Eval suite
Built 30 test cases. Initial pass rate: 62%. Primary failures: history filtering (agent ignored `history_check` results in 38% of cases), hallucinated URLs (33% of cases). Fixed by strengthening the system prompt and adding explicit output validation in the tool dispatcher. Pass rate after fixes: 90%.

### Week 4 — Observability + frontend
Instrumented with LangSmith. Mean latency: 14s. p95: 28s. Largest contributor: embedding generation during retrieval (~4s). Built the React frontend against the working API.

---

## Roadmap

### v1 — Complete
- [x] Custom agent loop (no LangChain)
- [x] RAG pipeline with hybrid retrieval
- [x] Eval suite with 30 test cases
- [x] LangSmith observability
- [x] FastAPI + WebSocket backend
- [x] React frontend, desktop + mobile

### v2 — Next
- [ ] Google Calendar integration (auto-block chosen plan)
- [ ] Learning from past turnout (weight scores by who actually showed up)
- [ ] Retrieval latency improvement — pre-filter by location before vector search
- [ ] Push notifications for RSVP updates

### v3 — Future
- [ ] Fine-tune a reranker on group preference data
- [ ] Public plan links shareable outside the app
- [ ] Group voting on plan finalists

---

*Built with the raw Anthropic API, pgvector, FastAPI, React, and Supabase. No LangChain.*
