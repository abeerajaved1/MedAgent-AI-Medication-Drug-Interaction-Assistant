# MedAgent

An AI Agent-based system for safe medication information and drug-drug
interaction checking. Implements the full pipeline:

```
User Query → Coordinator Agent → Intent + Tool Selection
           → RAG Agent / SQL Agent / Drug Interaction Tool
           → Pharmacist/Information Agent (draft)
           → Safety/Verifier Agent (approved / warning / insufficient_evidence)
           → Final structured response
```

> ⚠️ **This is a demo/educational system.** The knowledge base and
> interaction database contain a small, illustrative dataset — not a
> complete or clinically validated medical database. It is not a
> substitute for professional medical advice.

---

## 1. What's inside

| Component | File | Spec section |
|---|---|---|
| Coordinator Agent | `app/agents/coordinator.py` | 3.1 |
| RAG / Medication Info Agent | `app/agents/rag_agent.py`, `app/rag/knowledge_base.py` | 3.2 |
| SQL Database Agent | `app/agents/sql_agent.py`, `app/database.py` | 3.3 |
| Drug Interaction Tool | `app/agents/interaction_tool.py` | 3.4 |
| Pharmacist/Information Agent | `app/agents/pharmacist_agent.py` | 3.5 |
| Safety/Verifier Agent | `app/agents/safety_agent.py` | 3.6 |
| **External Source Agent** (new) | `app/agents/external_source_agent.py` | live web lookups |
| Orchestration / API | `app/main.py` | Section 5 workflow |
| Chat UI | `app/static/index.html` | — |

**RAG** uses TF-IDF (scikit-learn) instead of neural embeddings, so the
image stays small and runs comfortably on free-tier hosts (512MB RAM) with
no model downloads.

**External Source Agent** — so the system isn't limited to the local RAG
knowledge base and seed database, it also does *live* lookups against two
free, no-API-key-required public data sources on every `medicine_info` and
`drug_interaction` query:
- **RxNorm** (U.S. National Library of Medicine) — validates/normalizes
  the drug name against a real medical terminology database.
  `https://rxnav.nlm.nih.gov/REST`
- **openFDA Drug Label API** (U.S. FDA) — pulls indications, warnings,
  contraindications, adverse reactions, and (when present) the
  manufacturer's own drug-interactions text straight from the official
  product label. `https://api.fda.gov/drug/label.json`

Both are genuinely free and public — no signup, no key, no cost, generous
public rate limits. Every call has a short timeout and is wrapped so a
network hiccup or an unrecognized drug degrades gracefully to "not
found" instead of breaking the pipeline; the local RAG + SQL data still
answers the question even if these external services are unreachable.
Toggle with `EXTERNAL_SOURCES_ENABLED=true|false` in `.env`.
Retrieved external text is treated exactly like RAG/SQL evidence — it
still passes through the Pharmacist Agent's draft and the Safety Agent's
verification before reaching the user.

**LLM provider** is pluggable — set `LLM_PROVIDER=gemini` or `grok` in
`.env`. **If no API key is set, the app runs in a deterministic MOCK
mode** so you can test the entire multi-agent pipeline (routing, RAG, SQL,
interaction checking, and the rule-based safety overrides) for free with
zero setup, before wiring up a real key.

The Safety Agent has **hard rule-based overrides that cannot be bypassed
by the LLM**: any interaction with `severity = major` is always forced to
`warning` status, and any query with no matching evidence is always
forced to `insufficient_evidence` — regardless of what the LLM verification
step returns.

---

## 2. Run locally (no Docker)

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add GEMINI_API_KEY or GROK_API_KEY (or leave blank for mock mode)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** for the chat UI, or **http://localhost:8000/docs**
for the interactive API docs.

---

## 3. Run with Docker

```bash
cp .env.example .env
# edit .env with your API key (or leave blank to run in mock mode)

docker compose up --build
```

App is available at **http://localhost:8000**.

Data persists in the `medagent_data` Docker volume (the SQLite file), so
the database survives container restarts.

To stop: `docker compose down`

---

## 4. Getting a free LLM API key

**Option A — Gemini (recommended, generous free tier)**
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account and click "Create API key"
3. Put it in `.env` as `GEMINI_API_KEY=...`, set `LLM_PROVIDER=gemini`

**Option B — Grok (xAI)**
1. Go to https://console.x.ai and create an account
2. Generate an API key (xAI periodically offers free/trial credits)
3. Put it in `.env` as `GROK_API_KEY=...`, set `LLM_PROVIDER=grok`

Both are wired through `app/llm_client.py` — switching providers is just
changing `LLM_PROVIDER` in `.env`, no code changes needed.

---

## 5. Deploy for free

Any host that can run a Docker container works. Three good free options:

### Option A — Render.com (easiest)
1. Push this project to a GitHub repo.
2. On https://render.com → **New +** → **Web Service** → connect your repo.
3. Render auto-detects the `Dockerfile`. Choose the **Free** instance type.
4. Under **Environment**, add `LLM_PROVIDER`, `GEMINI_API_KEY` (or
   `GROK_API_KEY`), and `DATABASE_PATH=./data/medagent.db`.
5. Deploy. Render builds the Docker image and gives you a public URL.
   *(Free instances sleep after inactivity and cold-start on the next
   request — fine for a demo.)*

### Option B — Hugging Face Spaces (Docker SDK)
1. Create a new Space → SDK: **Docker**.
2. Push this repo's contents to the Space's git repo (it just needs the
   `Dockerfile` at the root, which it already has).
3. Add your API key under **Settings → Repository secrets**
   (`GEMINI_API_KEY`, `LLM_PROVIDER`).
4. The Space builds and serves the app automatically at
   `https://<you>-<space-name>.hf.space`.

### Option C — Fly.io
1. Install the `flyctl` CLI and run `fly launch` in this directory
   (it detects the Dockerfile automatically).
2. Set secrets: `fly secrets set LLM_PROVIDER=gemini GEMINI_API_KEY=xxx`
3. `fly deploy`
4. Fly's free allowance covers small always-on or scale-to-zero apps.

All three options are $0 as long as you stay within their free tier
limits (which this lightweight app — no GPU, no large models — comfortably does).

---

## 6. API reference

### `POST /api/chat`
```json
// Request
{ "message": "Can Warfarin and Aspirin be taken together?" }

// Response
{
  "query_type": "drug_interaction",
  "medicines": ["Warfarin", "Aspirin"],
  "interaction_status": "major",
  "explanation": "...",
  "safety_note": "...",
  "verification_status": "warning",
  "sources": [ { "source": "...", "snippet": "..." } ],
  "disclaimer": "..."
}
```

### `GET /api/health`
Returns `{"status": "ok", "llm_provider": "gemini"}`.

---

## 7. Extending the dataset

`data/knowledge_base.json` holds the RAG documents; `app/database.py`
holds the seed data for `MEDICINES`, `DRUG_INTERACTIONS`, and
`SAFETY_WARNINGS`. Add entries to either (and delete the SQLite file, or
your Docker volume, to force a reseed) to grow the demo dataset.

On top of the local data, the **External Source Agent** already pulls
live data from RxNorm and openFDA for every relevant query (see section 1
above) — so real medicines outside your seed dataset will still return
genuine indications/warnings/adverse-reactions text pulled from the FDA's
own label data, not just whatever you've hand-seeded locally. For a
production system handling real medical decisions, you'd still want a
licensed, clinically validated interaction database (e.g. a paid
DrugBank/Medi-Span/First Databank feed) rather than relying solely on
label-text cross-referencing.

---

## 8. Project layout

```
medagent/
├── app/
│   ├── main.py              # FastAPI app + pipeline orchestration
│   ├── config.py            # env-based settings
│   ├── database.py          # SQLite schema + seed + queries
│   ├── llm_client.py        # Gemini/Grok/mock LLM abstraction
│   ├── models.py            # Pydantic request/response models
│   ├── agents/
│   │   ├── coordinator.py
│   │   ├── rag_agent.py
│   │   ├── sql_agent.py
│   │   ├── interaction_tool.py
│   │   ├── pharmacist_agent.py
│   │   └── safety_agent.py
│   ├── rag/
│   │   └── knowledge_base.py  # TF-IDF retriever
│   └── static/
│       └── index.html         # chat UI
├── data/
│   └── knowledge_base.json    # RAG documents
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
