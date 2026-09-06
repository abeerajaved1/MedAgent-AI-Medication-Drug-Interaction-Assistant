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

## 2. Phase upgrades (v2)

Beyond the original 6-agent spec, this version adds three research-blueprint
upgrades, all still free:

**Phase 1 — real vector DB + eval harness**
- `app/rag/chroma_store.py`: Chroma (embedded, no server) + a free
  PubMedBERT sentence-transformers model (`pritamdeka/S-PubMedBert-MS-MARCO`)
  replace TF-IDF retrieval. If the embedding model can't be downloaded
  (e.g. fully offline build), it **automatically falls back to the
  original TF-IDF retriever** (`app/rag/knowledge_base.py`) — the app
  never hard-fails because of this.
- `eval/generate_eval_set.py`: bootstraps a ~130-question eval set
  (deterministic templates per medicine/interaction + LLM-paraphrased
  variants for phrasing robustness).
- `eval/run_eval.py`: runs the eval set through the real pipeline and
  reports retrieval hit rate, citation coverage, intent-classification
  accuracy, veto-loop frequency, and an LLM-judge groundedness/hallucination
  proxy. **Note:** the groundedness judge needs a real `GEMINI_API_KEY`/
  `GROK_API_KEY` to produce meaningful verdicts — in mock mode it reports
  `0.0` because there's no real model to judge with; the other metrics
  (retrieval, routing, veto rate) are real either way.

**Phase 3 — interaction graph + Safety Agent veto power**
- `app/graph/interaction_graph.py`: drug interactions are now a NetworkX
  graph, not a flat SQL row lookup — built from the local seed data by
  default, with an optional `load_twosides_csv()` loader for the free,
  public [TWOSIDES dataset](https://tatonettilab.org/offsides/) for much
  broader real-world interaction coverage (see docstring for the honest
  caveat: TWOSIDES gives statistical association scores, not
  clinician-graded severity, so severity is mapped via a documented
  heuristic).
- The Safety/Verifier Agent now has **real veto power**: if it doesn't
  approve a draft, the objection is sent back to the Pharmacist Agent for
  a bounded number of re-drafts (`MAX_VETO_ROUNDS`, default 1) instead of
  just appending a warning note to an unrevised draft. `revision_count`
  in the response tells you how many times this fired.

**Phase 5 — patient profile + polypharmacy checks**
- `app/database.py` gains a `patient_profiles` table (age, allergies,
  conditions, current medications) keyed by a `session_id` you choose.
- `app/agents/patient_profile_agent.py`: cross-checks the stored profile
  against whatever medicine is being discussed — allergy matches (direct
  name or via the medicine's own allergy warnings) and polypharmacy hits
  (does this medicine interact with anything already on the patient's
  medication list?) are surfaced as hard evidence the Safety Agent cannot
  downgrade.
- New endpoints: `POST /api/profile`, `GET /api/profile/{session_id}`.
  Pass `session_id` in `/api/chat` requests to apply the profile.
  **This means a single-medicine question with no mention of any other
  drug can still get flagged**, purely from stored session state — e.g.
  asking "Can I take Ibuprofen?" while the profile has Warfarin on the
  medication list correctly forces a `warning` even though Warfarin was
  never mentioned in that message.



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

## 3. Phase upgrades (v3): reasoning, vision, ablation

**Phase 2 — structured reasoning (no fine-tuning / no GPU needed)**
The original blueprint's SO3 called for QLoRA fine-tuning a 7-8B model to
teach pharmacist-style reasoning. That needs real GPU-hours and doesn't
belong in a free web app. Instead, `app/agents/reasoning_profile.py`
defines a structured chain-of-thought PROMPT TEMPLATE — age → allergies →
conditions → current medications → evidence check → recommendation →
warnings — injected into the Pharmacist Agent's system prompt on every
call. This is a legitimate prompted-reasoning technique, not a trained
model, and should be described that way in any write-up (a prompting
scaffold, not a fine-tuned reasoning adapter). It costs nothing extra
and works with whichever LLM provider you've already configured.

**Phase 4 — Vision Agent (preliminary chest X-ray observations)**
`app/agents/vision_agent.py` sends an uploaded image straight to
Gemini's multimodal endpoint (same free `GEMINI_API_KEY` you're already
using — no separate vision model to train or host) with a prompt that
strictly forbids diagnostic language ("diagnosis", "confirmed", "you
have X") and requires hedged, plain-language observations plus a
mandatory disclaimer. New endpoint:

```
POST /api/vision   (multipart form: file=<image>, session_id=<optional>)
```

If `session_id` is provided, the finding is stored and automatically
pulled into later `/api/chat` calls in the *same session* so the
Coordinator can synthesize imaging + pharmacological advice together —
e.g. upload an X-ray, then ask "what should I know about treating this,"
and the prior finding is included as evidence (clearly labeled
preliminary/unconfirmed). Vision analysis only works with
`LLM_PROVIDER=gemini` and a real key — Grok's public API doesn't
currently expose a free vision endpoint, so with Grok (or no key) this
endpoint returns a clear "unavailable" response rather than guessing.

**Phase 6 — ablation study harness**
```bash
python -m eval.run_ablation           # full eval set, 4 configurations
python -m eval.run_ablation --quick   # first 20 cases, for a fast sanity check
```
Runs the same eval set through 4 configurations — baseline (local RAG +
SQL + interaction graph only) → +external sources → +safety veto loop →
+patient profile (full system) — toggling settings at runtime, and
prints a comparison table plus `eval/ablation_results.json`. This is
what SO7 ("quantify the marginal contribution of each component") asks
for. Note: the patient-profile row is rule-based and shows a real signal
even in mock mode; the external-sources row needs real internet access
(it's blocked in fully offline/sandboxed builds) to show its true delta.

---

## 4. Novelty additions (v4): multilingual, confidence, feedback loop, WHO EML, baseline & robustness evaluation

These close the remaining gaps between the original project proposal and
what a research paper needs, all free, no fine-tuning.

**Multilingual output** — `app/agents/pharmacist_agent.py` instructs the
LLM to answer in the same language the question was asked in. This is
generation-side only: the knowledge base and retrieval stay English, so
retrieval quality for non-English queries depends on the LLM's
cross-lingual understanding of the query, not true multilingual
retrieval — document this distinction honestly.

**Numeric confidence score** — every `/api/chat` response now includes
`confidence_score` (0.0-1.0), a transparent heuristic combining
retrieval similarity, whether the interaction graph gave a concrete
answer, and the Safety Agent's verdict as a hard ceiling (a "warning"
response can never report high confidence, even with strong retrieval).
This is explicitly **not** a calibrated model probability — say so in
the paper. See `_compute_confidence()` in `app/pipeline.py`.

**Feedback loop ("continual improvement without full retraining")** —
`POST /api/feedback` (rating: `helpful`/`not_helpful`, optional
`source_ref_ids` from a response's `sources[].ref_id`) and
`GET /api/feedback/stats`. Negative feedback on a specific RAG document
nudges its retrieval trust multiplier down (and positive feedback nudges
it up), applied as a re-ranking factor in both `app/rag/chroma_store.py`
and `app/rag/knowledge_base.py`. This is a real, working mechanism, but
an intentionally simple one: it reweights existing documents, it does
not learn new facts or retrain anything.

**WHO Essential Medicines List tagging** — the `medicines` table now has
`on_who_eml`, tagging whether each seeded medicine (or its drug-class
representative) is on the WHO Model List of Essential Medicines, 23rd
list (2023) — verified by web search, not guessed (Sertraline and
Lisinopril are marked `False` because the EML's representative drugs
for SSRIs/ACE-inhibitors are Fluoxetine/Enalapril, not these specific
agents). This is an illustrative "global relevance" signal for the
paper's regional-availability discussion, not a full formulary system —
say so explicitly rather than overclaiming international coverage.

**Baseline comparison (the RQ1 headline result)**
```bash
python -m eval.run_baseline_comparison         # full eval set
python -m eval.run_baseline_comparison --quick # interaction-only subset, capped
```
Runs the same questions through (1) the raw configured LLM with zero
tools/evidence/safety-layer, and (2) the full MedAgent pipeline, and
reports citation rate and caution-surfaced rate for each. In testing
this produced a real, meaningful gap (`medagent caution_rate: 1.0` vs
`raw_llm caution_rate: 0.0` on interaction questions) — this is the
single most important number for justifying the multi-agent approach
over a monolithic LLM. Writes `eval/baseline_comparison_results.json`.

**Safety-consistency / robustness audit (scoped deliberately, not a
demographic bias audit)**
```bash
python -m eval.run_robustness_audit
```
Tests the same questions across 5 patient sub-groups (pediatric, adult
control, elderly, renal impairment, polypharmacy-on-Warfarin) and
reports where `verification_status` diverges from the adult control —
divergence that matches a real clinical basis in the evidence
(polypharmacy-on-Warfarin correctly flagging on Ibuprofen/Aspirin but
not Metformin) is the system working correctly; divergence with no
basis would flag an inconsistency. This is scoped to age/comorbidity
rather than race/ethnicity/gender **on purpose**: the system never
collects those attributes (by design — see its own privacy handling),
so there's no demographic-labeled data to audit fairness across in the
first place. State this scoping decision explicitly in the paper's
limitations section rather than silently omitting a bias audit. Writes
`eval/robustness_audit_results.json`.

---

## 6. Run with Docker

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

## 7. Getting a free LLM API key

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

## 8. Deploy for free

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

## 9. API reference

### `POST /api/chat`
```json
// Request
{ "message": "Can Warfarin and Aspirin be taken together?", "session_id": "optional-user-id" }

// Response
{
  "query_type": "drug_interaction",
  "medicines": ["Warfarin", "Aspirin"],
  "interaction_status": "major",
  "explanation": "...",
  "safety_note": "...",
  "verification_status": "warning",
  "confidence_score": 0.7,
  "revision_count": 0,
  "profile_flags": [],
  "sources": [ { "source": "...", "snippet": "...", "ref_id": "warfarin_2" } ],
  "disclaimer": "..."
}
```

### `POST /api/profile`
```json
{ "session_id": "user-123", "age": 45, "allergies": ["penicillin"],
  "conditions": ["hypertension"], "current_medications": ["Warfarin"] }
```

### `GET /api/profile/{session_id}`
Returns the stored profile, or 404.

### `POST /api/vision`
Multipart form: `file=<image>`, optional `session_id`. Returns
`{"available": bool, "findings": str, "disclaimer": str, "saved_to_session": bool}`.
Requires `LLM_PROVIDER=gemini` with a real key — otherwise returns
`available: false` with an explanatory message.

### `POST /api/feedback`
```json
{ "query": "What is Ibuprofen used for?", "rating": "not_helpful",
  "query_type": "medicine_info", "source_ref_ids": ["ibuprofen_1"] }
```
`rating` must be `"helpful"` or `"not_helpful"`. `source_ref_ids` should
be `ref_id` values copied from that response's `sources[]` — only RAG
documents have a `ref_id` (non-RAG sources return `null` and can't be
targeted by feedback). Returns `{"status": "recorded"}`.

### `GET /api/feedback/stats`
Returns aggregate counts: `{"total": int, "helpful": int, "not_helpful": int, "by_query_type": {...}}`.

### `GET /api/health`
Returns `{"status": "ok", "llm_provider": "gemini"}`.

---

## 9a. Running the evaluation harness

```bash
python -m eval.generate_eval_set             # writes eval/eval_set.json (~130 cases)
python -m eval.run_eval                      # writes eval/eval_results.json + prints a summary
python -m eval.run_ablation                  # component ablation (4 configs)
python -m eval.run_baseline_comparison       # raw LLM vs full MedAgent (RQ1 headline result)
python -m eval.run_robustness_audit          # safety-consistency across patient sub-groups
```

Re-run `generate_eval_set` any time you add medicines/interactions to
`app/database.py` to keep the eval set in sync. **All of the above give
meaningful structural results even in mock mode (no API key) for the
rule-based components (routing, retrieval, interaction graph, profile
checks), but the LLM-judged metrics (groundedness, raw-LLM baseline
text) need a real `GEMINI_API_KEY`/`GROK_API_KEY` to produce numbers
worth putting in a paper.**

---

## 10. Extending the dataset

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

## 11. Project layout

```
medagent/
├── app/
│   ├── main.py                 # FastAPI routes only (thin)
│   ├── pipeline.py             # orchestration: routing → tools → veto loop → response
│   ├── config.py               # env-based settings
│   ├── database.py             # SQLite: medicines/interactions/warnings/patient_profiles/feedback/trust
│   ├── llm_client.py           # Gemini/Grok/mock LLM abstraction
│   ├── models.py                # Pydantic request/response models
│   ├── agents/
│   │   ├── coordinator.py
│   │   ├── rag_agent.py
│   │   ├── sql_agent.py
│   │   ├── interaction_tool.py       # now graph-backed (Phase 3)
│   │   ├── external_source_agent.py  # RxNorm + openFDA live lookups
│   │   ├── patient_profile_agent.py  # allergy + polypharmacy checks (Phase 5)
│   │   ├── reasoning_profile.py      # structured CoT prompt template + multilingual instruction (Phase 2 / v4)
│   │   ├── vision_agent.py           # chest X-ray via Gemini multimodal (Phase 4)
│   │   ├── pharmacist_agent.py       # draft() + revised draft on veto (Phase 3)
│   │   └── safety_agent.py
│   ├── graph/
│   │   └── interaction_graph.py      # NetworkX graph + optional TWOSIDES loader (Phase 3)
│   ├── rag/
│   │   ├── chroma_store.py           # Chroma + PubMedBERT + trust-score re-ranking (Phase 1 / v4)
│   │   └── knowledge_base.py         # TF-IDF fallback, same trust-score re-ranking
│   └── static/
│       └── index.html
├── eval/
│   ├── generate_eval_set.py         # Phase 1: bootstraps eval_set.json
│   ├── run_eval.py                  # Phase 1: runs eval_set.json, reports metrics
│   ├── run_ablation.py              # Phase 6: component ablation study
│   ├── run_baseline_comparison.py   # v4: raw LLM vs full MedAgent (RQ1 result)
│   ├── run_robustness_audit.py      # v4: safety-consistency across patient sub-groups
│   ├── eval_set.json                # generated
│   └── eval_results.json            # generated
├── data/
│   ├── knowledge_base.json     # RAG documents
│   └── chroma/                 # Chroma's on-disk index (generated, gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
