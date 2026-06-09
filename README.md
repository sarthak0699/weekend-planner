# WeekendAI — Design Document

> A multi-agent AI system that automates weekend planning for friend groups. It finds events, curates them to group interests, and handles invites — so one person doesn't have to do all the organizing.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Agent System](#agent-system)
5. [Tech Stack](#tech-stack)
6. [Data Models](#data-models)
7. [API Design](#api-design)
8. [Frontend](#frontend)
9. [Folder Structure](#folder-structure)
10. [Environment Variables](#environment-variables)
11. [Getting Started](#getting-started)
12. [Roadmap](#roadmap)

---

## Overview

WeekendAI is a full-stack application built around a **LangGraph multi-agent pipeline**. A user sets up their profile and friend group once. Every week, three specialized AI agents collaborate to:

1. Search for events matching each friend's interests
2. Curate and rank plans based on group compatibility and recent history
3. Send personalized invites and collect RSVPs

The app is designed to work on both desktop and mobile, with real-time agent status streamed to the UI via WebSocket.

---

## Problem Statement

Weekend planning in a friend group usually falls on one person. They manually search for things to do, try to account for everyone's preferences, message each person individually, and chase RSVPs. This is tedious and often leads to either doing the same things repeatedly or low turnout because coordination broke down.

**WeekendAI automates the coordination layer** — not just the search, but the full loop from discovery to confirmed plans.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    React Frontend                    │
│         (Vite + Tailwind, desktop + mobile)          │
└───────────────────┬─────────────────────────────────┘
                    │ REST + WebSocket
┌───────────────────▼─────────────────────────────────┐
│                  FastAPI Backend                      │
│           (Python, async, WebSocket support)          │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│              LangGraph Supervisor                     │
│         (orchestrates the 3 agents below)            │
├──────────────┬──────────────┬───────────────────────┤
│ Search Agent │ Planner Agent│   Messaging Agent      │
│  (Tavily)    │  (Claude)    │  (Twilio / Gmail)      │
└──────────────┴──────────────┴───────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                   Supabase                            │
│       (users, friends, interests, event history)     │
└─────────────────────────────────────────────────────┘
```

### Key design decisions

**Why LangGraph over a simple chain?**
The planning loop is not linear. The planner agent may decide the initial search results are too thin for a given weekend and trigger the search agent again with refined parameters. LangGraph's conditional edges and shared state make this kind of feedback loop clean to implement and easy to extend.

**Why WebSocket for agent status?**
The full agent pipeline takes 10–30 seconds. Streaming agent activity logs in real time keeps the UI responsive and gives users visibility into what's happening, which builds trust in the output.

**Why Supabase?**
Built-in auth, a Postgres database, and a JavaScript/Python SDK. The `event_history` table is the most important: it's what lets the planner agent avoid recommending things the group did recently.

---

## Agent System

The three agents run inside a **LangGraph StateGraph**. They share a single `WeekendState` object that is passed between nodes and updated at each step.

### Shared state

```python
class WeekendState(TypedDict):
    user_id: str
    location: str
    weekend_dates: list[str]
    friends: list[FriendProfile]
    combined_interests: list[str]
    recent_activities: list[str]       # from event_history table
    raw_search_results: list[Event]    # output of Search Agent
    curated_plans: list[Plan]          # output of Planner Agent
    invite_status: dict[str, str]      # output of Messaging Agent
    agent_logs: list[str]              # streamed to frontend
```

### Search Agent

**Responsibility:** Discover relevant events for the upcoming weekend.

**How it works:**
- Receives `combined_interests`, `location`, and `weekend_dates` from state
- Runs parallel Tavily searches — one query per interest category
- Deduplicates results and normalizes them into `Event` objects
- Writes `raw_search_results` back to state

**Tools:** `tavily_search`, `supabase_get_history`

**Example queries generated:**
```
"hiking trails near San Francisco this weekend"
"live music San Francisco June 2025"
"food festivals San Francisco June 14-15"
```

### Planner Agent

**Responsibility:** Curate and rank raw search results into 3–5 recommended plans.

**How it works:**
- Reads `raw_search_results` and `recent_activities` from state
- Filters out events the group did in the past 4 weeks
- Scores each event against each friend's interest profile
- Computes a **group compatibility score** (average of individual scores, penalized for low-interest outliers)
- Returns top plans sorted by score, with registration links attached
- Uses a conditional edge: if fewer than 3 plans score above the threshold, triggers the Search Agent again with a broader query

**Scoring formula:**
```
group_score = mean(individual_scores) * (1 - outlier_penalty)

where outlier_penalty = stdev(individual_scores) / 100
```

This rewards plans that most people are genuinely excited about over plans where one person loves it and others are indifferent.

### Messaging Agent

**Responsibility:** Send personalized invites to each friend and track RSVPs.

**How it works:**
- Takes the user-selected plan from `curated_plans`
- Generates a personalized message per friend (referencing their specific interests that match the event)
- Sends via Twilio SMS or Gmail depending on user preference
- Writes an RSVP tracking record to Supabase
- Polls for replies and updates `invite_status` in state

**Tools:** `twilio_send_sms`, `gmail_send`, `supabase_write_rsvp`

### Supervisor / Graph definition

```python
from langgraph.graph import StateGraph, END
from agents import search_agent, planner_agent, messaging_agent

def route_after_planning(state: WeekendState):
    high_quality = [p for p in state["curated_plans"] if p["score"] >= 70]
    if len(high_quality) < 3:
        return "search"   # loop back, refine
    return "message"      # proceed to invites

graph = StateGraph(WeekendState)
graph.add_node("search",  search_agent)
graph.add_node("plan",    planner_agent)
graph.add_node("message", messaging_agent)

graph.set_entry_point("search")
graph.add_edge("search", "plan")
graph.add_conditional_edges("plan", route_after_planning, {
    "search": "search",
    "message": "message"
})
graph.add_edge("message", END)

app = graph.compile()
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React + Vite + Tailwind | Fast builds, responsive by default |
| Backend | FastAPI + Python | Async support, easy WebSocket, great LangChain ecosystem |
| Agent framework | LangGraph | Stateful multi-agent graphs with conditional edges |
| LLM | Claude claude-sonnet-4-20250514 (Anthropic) | Strong reasoning for planning and personalization |
| Event search | Tavily API | LangChain-native, built for agent tool use |
| Database + Auth | Supabase | Auth + Postgres + SDK, minimal ops overhead |
| Messaging | Twilio (SMS) / Gmail API | Broad reach, easy RSVP tracking |
| Realtime | WebSocket (FastAPI native) | Stream agent logs to frontend as they happen |
| Deployment | Docker + docker-compose | Consistent local dev and production parity |

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
  category text not null   -- e.g. "hiking", "live music", "food"
);

-- Friend groups
create table friendships (
  user_id    uuid references users(id) on delete cascade,
  friend_id  uuid references users(id) on delete cascade,
  primary key (user_id, friend_id)
);

-- Events (normalized output from Search Agent)
create table events (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  description text,
  category    text,
  location    text,
  event_date  date,
  url         text,
  price       text,
  created_at  timestamptz default now()
);

-- Event history (what a group has done)
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
  status     text default 'pending',  -- pending | yes | no | maybe
  sent_at    timestamptz default now(),
  updated_at timestamptz
);
```

### Event object (Python)

```python
class Event(BaseModel):
    name: str
    description: str
    category: str
    location: str
    event_date: str
    url: str | None
    price: str | None
    score: float | None = None          # set by Planner Agent
    matched_interests: list[str] = []   # which friend interests it hits
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

POST   /plans/generate           Kick off the agent pipeline (returns plan_id)
GET    /plans/{plan_id}          Get curated plans for a session
POST   /plans/{plan_id}/select   User selects a plan → triggers Messaging Agent

GET    /rsvps/{plan_id}          Get RSVP status for a plan
GET    /history                  Get past events
```

### WebSocket

```
WS  /ws/{plan_id}
```

Streams agent log events as JSON while the pipeline runs:

```json
{ "agent": "search",  "status": "running", "message": "Searching for hiking events near SF..." }
{ "agent": "search",  "status": "done",    "message": "Found 18 events across 4 categories" }
{ "agent": "plan",    "status": "running", "message": "Scoring events against group interests..." }
{ "agent": "plan",    "status": "done",    "message": "4 plans ready, top score 92%" }
```

---

## Frontend

### Pages

| Route | Description |
|---|---|
| `/` | Dashboard — current weekend plans, agent status, friend RSVPs |
| `/setup` | Onboarding — set location, add friends, pick interests |
| `/friends` | Manage friend group |
| `/history` | Past weekends |
| `/settings` | Notifications, messaging preferences |

### Key components

```
src/
├── components/
│   ├── AgentLog.tsx          # Live streaming agent activity feed
│   ├── EventCard.tsx         # Plan card with match score + register link
│   ├── FriendList.tsx        # Sidebar friend group with RSVP status
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

Tailwind breakpoints are used throughout. The sidebar collapses to a bottom nav on mobile. Event cards switch from a 2-column grid to a single column below `sm`. The agent log is hidden by default on mobile and accessible via a slide-up drawer.

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
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── search_agent.py
│   │   ├── planner_agent.py
│   │   └── messaging_agent.py
│   ├── tools/
│   │   ├── tavily_tools.py
│   │   ├── supabase_tools.py
│   │   └── messaging_tools.py
│   ├── graph.py              # LangGraph supervisor definition
│   ├── state.py              # WeekendState TypedDict
│   ├── models.py             # Pydantic models
│   ├── routers/
│   │   ├── plans.py
│   │   ├── users.py
│   │   └── friends.py
│   ├── ws.py                 # WebSocket handler
│   └── main.py               # FastAPI app entry point
│
├── supabase/
│   └── schema.sql
│
├── docker-compose.yml
├── .env.example
├── DESIGN.md                 # this file
└── README.md
```

---

## Environment Variables

```bash
# .env.example

# Anthropic
ANTHROPIC_API_KEY=

# Tavily
TAVILY_API_KEY=

# Supabase
SUPABASE_URL=
SUPABASE_SERVICE_KEY=

# Twilio (optional — for SMS invites)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=

# Gmail (optional — for email invites)
GMAIL_CLIENT_ID=
GMAIL_CLIENT_SECRET=
GMAIL_REFRESH_TOKEN=

# App
FRONTEND_URL=http://localhost:5173
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional but recommended)

### Local setup

```bash
# Clone the repo
git clone https://github.com/yourusername/weekend-planner
cd weekend-planner

# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env    # fill in your keys
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### With Docker

```bash
cp .env.example .env   # fill in your keys
docker-compose up
```

Frontend runs on `http://localhost:5173`, backend on `http://localhost:8000`.

---

## Roadmap

### v1 — Core (current scope)
- [x] LangGraph multi-agent pipeline
- [x] Tavily event search with interest-based queries
- [x] Group compatibility scoring
- [x] Supabase schema and auth
- [x] FastAPI + WebSocket backend
- [x] React frontend, desktop + mobile

### v2 — Improvements
- [ ] Google Calendar integration (auto-block the chosen plan)
- [ ] Push notifications for RSVP updates
- [ ] Learning from past turnout (weight scores by who actually showed up)
- [ ] Support for multiple simultaneous plans (Saturday plan + Sunday plan)

### v3 — Social layer
- [ ] Public plan links shareable outside the app
- [ ] Polls — let friends vote on finalists before invites go out
- [ ] Group chat within a plan session

---

*Built with LangGraph, LangChain, FastAPI, React, and Supabase.*
