"""
Central configuration for MedAgent.
Reads everything from environment variables (.env in local/docker runs).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-2-latest")
    GROK_BASE_URL: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "./data/medagent.db")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    EXTERNAL_SOURCES_ENABLED: bool = os.getenv("EXTERNAL_SOURCES_ENABLED", "true").lower() == "true"
    EXTERNAL_TIMEOUT_SECONDS: float = float(os.getenv("EXTERNAL_TIMEOUT_SECONDS", "6"))

    # Phase 1: Chroma vector DB + medical embeddings
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "pritamdeka/S-PubMedBert-MS-MARCO")

    # Phase 3: safety veto / re-draft loop
    MAX_VETO_ROUNDS: int = int(os.getenv("MAX_VETO_ROUNDS", "1"))

    # Phase 4: Vision Agent (Gemini multimodal, free tier)
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "gemini-2.0-flash")

    # --- New capability toggles (all default ON; each degrades gracefully) ---

    # Emergency/Triage Agent: red-flag short-circuit before the normal pipeline.
    EMERGENCY_AGENT_ENABLED: bool = os.getenv("EMERGENCY_AGENT_ENABLED", "true").lower() == "true"

    # Adaptive clarification: ask a follow-up question instead of guessing
    # when safety-relevant info is missing.
    CLARIFICATION_AGENT_ENABLED: bool = os.getenv("CLARIFICATION_AGENT_ENABLED", "true").lower() == "true"

    # Multi-turn conversational memory: how many prior turns to inject as context.
    CONVERSATION_MEMORY_ENABLED: bool = os.getenv("CONVERSATION_MEMORY_ENABLED", "true").lower() == "true"
    CONVERSATION_MEMORY_TURNS: int = int(os.getenv("CONVERSATION_MEMORY_TURNS", "4"))

    # Self-consistency check: draft twice, flag disagreement. Costs one extra
    # LLM call per query, so it's opt-in-by-default-on but easy to disable
    # for cost-sensitive deployments.
    SELF_CONSISTENCY_ENABLED: bool = os.getenv("SELF_CONSISTENCY_ENABLED", "true").lower() == "true"

    # Sentence-level citation attribution (TF-IDF based, no extra LLM cost).
    CITATION_ATTRIBUTION_ENABLED: bool = os.getenv("CITATION_ATTRIBUTION_ENABLED", "true").lower() == "true"

    # Full reasoning-trace export, retrievable via GET /api/trace/{query_id}.
    REASONING_TRACE_ENABLED: bool = os.getenv("REASONING_TRACE_ENABLED", "true").lower() == "true"

    # Brand-to-generic name resolution via RxNorm (piggybacks on
    # EXTERNAL_SOURCES_ENABLED / EXTERNAL_TIMEOUT_SECONDS above).
    NAME_RESOLUTION_ENABLED: bool = os.getenv("NAME_RESOLUTION_ENABLED", "true").lower() == "true"

    # Escalate-to-human trigger threshold (see app/agents/escalation_agent.py).
    ESCALATION_CONFIDENCE_THRESHOLD: float = float(os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", "0.35"))

    # Tier 1 rule-based dosing safety checks: age-based dosing rules,
    # pregnancy/breastfeeding contraindications, renal/hepatic impairment
    # rules, and dosage boundary checking (see app/agents/dosing_safety_agent.py).
    DOSING_SAFETY_ENABLED: bool = os.getenv("DOSING_SAFETY_ENABLED", "true").lower() == "true"

    # Tier 2: Vision Agent -> Symptom Checker Agent -> Pharmacist Agent chain.
    # When a stored Vision Agent finding exists for a session, route it
    # through the Symptom Checker Agent (as if it were a reported symptom)
    # before folding into the Pharmacist Agent's evidence (see app/pipeline.py).
    VISION_SYMPTOM_CHAIN_ENABLED: bool = os.getenv("VISION_SYMPTOM_CHAIN_ENABLED", "true").lower() == "true"

    # Tier 3 #7: cross-encoder reranking (see app/rag/reranker.py). Free,
    # no fine-tuning — an extra CPU inference pass with a small, public
    # sentence-transformers cross-encoder model, over-fetching candidates
    # from the base retriever and re-scoring them for a more accurate
    # final top-K. Fails soft (falls back to the base retriever's own
    # ranking) if the model can't be loaded, e.g. no internet access at
    # first run — same resilience pattern as the Chroma embedding model in
    # app/rag/chroma_store.py.
    RERANKING_ENABLED: bool = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_OVER_FETCH_MULTIPLIER: int = int(os.getenv("RERANK_OVER_FETCH_MULTIPLIER", "3"))

    # Tier 5 #10: "block vs warn" behavior for major interactions/dosing
    # concerns. Default false preserves the existing behavior (show the
    # answer with a forced warning + escalation flag). Set true to instead
    # withhold the drafted explanation entirely in favor of a short,
    # safety-only message when a major concern is found (see app/pipeline.py).
    BLOCK_ON_MAJOR_SEVERITY: bool = os.getenv("BLOCK_ON_MAJOR_SEVERITY", "false").lower() == "true"

    # Tier 3 #8: optional real TWOSIDES data load at startup. Unset by
    # default (the free public dataset must be downloaded by you — see
    # scripts/load_twosides.py for why this can't happen automatically).
    # When set, app/main.py's startup hook merges this CSV/csv.gz into the
    # interaction graph every time the app starts, so the extra coverage
    # persists across restarts without re-running the script manually.
    TWOSIDES_CSV_PATH: str = os.getenv("TWOSIDES_CSV_PATH", "")
    TWOSIDES_MAX_ROWS: int | None = (
        int(os.getenv("TWOSIDES_MAX_ROWS")) if os.getenv("TWOSIDES_MAX_ROWS") else None
    )


settings = Settings()
