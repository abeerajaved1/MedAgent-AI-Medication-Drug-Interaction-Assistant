"""
SQLite database layer: schema creation + seed data + query helpers
for MEDICINES, DRUG_INTERACTIONS, SAFETY_WARNINGS.

New in this revision (see the new agent capabilities under
app/agents/ and app/trace/):
  - `conversation_turns` table + save_conversation_turn / get_recent_conversation_turns
    for multi-turn conversational memory.
  - `reasoning_traces` table + save_trace / get_trace for the full
    reasoning-trace export feature (GET /api/trace/{query_id}).
"""
import sqlite3
import os
import logging
from contextlib import contextmanager
from typing import Optional

from app.config import settings

logger = logging.getLogger("medagent.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS medicines (
    medicine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    generic_name TEXT,
    category TEXT,
    on_who_eml INTEGER DEFAULT 0   -- WHO Essential Medicines List flag (illustrative — see SEED_MEDICINES note)
);

CREATE TABLE IF NOT EXISTS drug_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug1 TEXT NOT NULL,
    drug2 TEXT NOT NULL,
    severity TEXT NOT NULL,           -- minor | moderate | major
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS safety_warnings (
    warning_id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine TEXT NOT NULL,
    warning_type TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    session_id TEXT PRIMARY KEY,
    age INTEGER,
    allergies TEXT,        -- comma-separated
    conditions TEXT,       -- comma-separated
    current_medications TEXT,  -- comma-separated
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_findings (
    finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    finding_text TEXT NOT NULL,
    source TEXT DEFAULT 'vision_agent',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    query TEXT NOT NULL,
    query_type TEXT,
    rating TEXT NOT NULL,          -- 'helpful' | 'not_helpful'
    comment TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS doc_trust_scores (
    doc_id TEXT PRIMARY KEY,
    trust_multiplier REAL NOT NULL DEFAULT 1.0,
    positive_count INTEGER NOT NULL DEFAULT 0,
    negative_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,            -- 'user' | 'assistant'
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
    ON conversation_turns (session_id, turn_id);

CREATE TABLE IF NOT EXISTS reasoning_traces (
    query_id TEXT PRIMARY KEY,
    session_id TEXT,
    query_text TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS history_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    item_type TEXT NOT NULL,        -- medication | interaction | symptoms | image_analysis | prescription
    title TEXT NOT NULL,
    preview TEXT NOT NULL,
    confidence REAL,
    verification_status TEXT,
    saved INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT,              -- full structured result, for the "view" action
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_history_items_session
    ON history_items (session_id, created_at DESC);

-- Tier 1 rule-based safety tables (see app/agents/dosing_safety_agent.py).
-- All four are small, hand-curated illustrative rule sets for the demo
-- medicine list — NOT a complete or clinically validated dosing reference.
-- Real deployments should source these from a licensed drug database.

CREATE TABLE IF NOT EXISTS age_dosing_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine TEXT NOT NULL,
    min_age INTEGER,               -- inclusive lower bound in years; NULL = no lower bound
    max_age INTEGER,               -- inclusive upper bound in years; NULL = no upper bound
    rule_type TEXT NOT NULL,       -- avoid | caution | dose_adjust
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pregnancy_rules (
    medicine TEXT PRIMARY KEY,
    pregnancy_category TEXT NOT NULL,     -- safe | caution | avoid | unknown
    pregnancy_note TEXT,
    breastfeeding_category TEXT NOT NULL, -- safe | caution | avoid | unknown
    breastfeeding_note TEXT
);

CREATE TABLE IF NOT EXISTS organ_impairment_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine TEXT NOT NULL,
    organ TEXT NOT NULL,           -- renal | hepatic
    rule_type TEXT NOT NULL,       -- avoid | dose_adjust | monitor
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medicine_dosage_limits (
    medicine TEXT PRIMARY KEY,
    unit_dose_mg REAL,             -- typical mg per tablet/capsule, if applicable
    max_daily_mg REAL,             -- adult max daily dose in mg; NULL = individualized (e.g. Warfarin)
    notes TEXT
);
"""

# NOTE: This is a small illustrative dataset for demo purposes only.
# It is NOT a complete or clinically validated drug interaction database.
#
# on_who_eml: whether the medicine (or its drug class representative) appears
# on the WHO Model List of Essential Medicines, 23rd list (2023) — a free,
# public reference (https://list.essentialmeds.org). Verified by web search
# on 2026-08-25; Sertraline and Lisinopril are marked False because the EML's
# representative drugs for their classes are Fluoxetine (SSRI) and Enalapril
# (ACE inhibitor) respectively, not these specific agents — this is an
# illustrative "global relevance" tag for demo purposes, not a substitute
# for checking the official list.
SEED_MEDICINES = [
    ("Paracetamol", "Acetaminophen", "Analgesic / Antipyretic", True),
    ("Ibuprofen", "Ibuprofen", "NSAID", True),
    ("Aspirin", "Acetylsalicylic acid", "NSAID / Antiplatelet", True),
    ("Warfarin", "Warfarin sodium", "Anticoagulant", True),
    ("Metformin", "Metformin HCl", "Antidiabetic (Biguanide)", True),
    ("Amoxicillin", "Amoxicillin", "Antibiotic (Penicillin)", True),
    ("Omeprazole", "Omeprazole", "Proton Pump Inhibitor", True),
    ("Sertraline", "Sertraline HCl", "SSRI Antidepressant", False),
    ("Simvastatin", "Simvastatin", "Statin", True),
    ("Lisinopril", "Lisinopril", "ACE Inhibitor", False),
]

SEED_INTERACTIONS = [
    ("Warfarin", "Aspirin", "major",
     "Combining Warfarin with Aspirin significantly increases the risk of "
     "bleeding because both affect blood clotting through different mechanisms."),
    ("Warfarin", "Ibuprofen", "major",
     "NSAIDs like Ibuprofen can increase bleeding risk when combined with "
     "Warfarin and may also reduce Warfarin's clearance."),
    ("Aspirin", "Ibuprofen", "moderate",
     "Ibuprofen may interfere with the antiplatelet effect of low-dose Aspirin "
     "if taken too close together."),
    ("Sertraline", "Aspirin", "moderate",
     "SSRIs combined with NSAIDs/Aspirin may increase the risk of "
     "gastrointestinal bleeding."),
    ("Simvastatin", "Amoxicillin", "minor",
     "No major interaction typically expected, but monitor for unusual "
     "muscle symptoms as a general precaution with statins."),
    ("Omeprazole", "Metformin", "minor",
     "No clinically significant interaction commonly reported; "
     "monitor as part of routine care."),
    ("Lisinopril", "Ibuprofen", "moderate",
     "NSAIDs may reduce the blood-pressure-lowering effect of ACE inhibitors "
     "like Lisinopril and can affect kidney function, especially with "
     "prolonged use."),
]

SEED_WARNINGS = [
    ("Warfarin", "Bleeding Risk",
     "Requires regular INR monitoring; risk of serious bleeding increases with "
     "many other medications, alcohol, and certain foods (e.g. vitamin K rich)."),
    ("Ibuprofen", "GI / Cardiovascular",
     "May cause stomach irritation, ulcers, or increased cardiovascular risk "
     "with long-term use, especially in older adults."),
    ("Aspirin", "Reye's Syndrome",
     "Should not be given to children or teenagers with viral infections due "
     "to the risk of Reye's syndrome."),
    ("Metformin", "Lactic Acidosis",
     "Rare but serious risk of lactic acidosis, particularly in patients with "
     "kidney impairment."),
    ("Sertraline", "Serotonin Syndrome",
     "Risk increases when combined with other serotonergic drugs; watch for "
     "agitation, rapid heartbeat, and high fever."),
    ("Simvastatin", "Muscle Toxicity",
     "May cause myopathy or, rarely, rhabdomyolysis; report unexplained "
     "muscle pain or weakness immediately."),
    ("Amoxicillin", "Allergic Reaction",
     "Contraindicated in patients with a known penicillin allergy; can cause "
     "reactions ranging from rash to anaphylaxis."),
]

# --- Tier 1 rule-based safety seed data -----------------------------------
# Hand-curated, illustrative only (same caveat as SEED_MEDICINES/SEED_WARNINGS
# above) — sourced from general public drug-label knowledge for demo purposes,
# not a substitute for a licensed clinical drug-safety database. Ages are in
# whole years; "elderly" cautions use min_age with no max_age.

SEED_AGE_DOSING_RULES = [
    ("Aspirin", 0, 18, "avoid",
     "Avoid in children and teenagers, especially during viral illness, due to the risk of Reye's syndrome."),
    ("Ibuprofen", 0, 0.5, "caution",
     "Use in infants under 6 months should only be under medical supervision."),
    ("Ibuprofen", 65, None, "caution",
     "Older adults have higher risk of GI bleeding and kidney effects — use the lowest effective dose."),
    ("Aspirin", 65, None, "caution",
     "Older adults have higher risk of GI bleeding with regular use."),
    ("Warfarin", 65, None, "caution",
     "Older adults are typically more sensitive to Warfarin and often need lower doses with closer INR monitoring."),
    ("Sertraline", 0, 24, "caution",
     "Antidepressants carry an FDA boxed warning for increased suicidal thoughts in children, teens, and young adults."),
    ("Metformin", 0, 10, "caution",
     "Use in children under 10 is not well established — pediatric dosing should be supervised by a specialist."),
    ("Simvastatin", 0, 10, "caution",
     "Use in children under 10 is not well established."),
    ("Omeprazole", 0, 1, "caution",
     "Use in infants under 1 year should only be under medical supervision."),
    ("Lisinopril", 0, 6, "caution",
     "Safety and dosing in children under 6 are not well established."),
]

SEED_PREGNANCY_RULES = [
    ("Paracetamol", "safe", "Generally considered safe throughout pregnancy at recommended doses.",
     "safe", "Considered compatible with breastfeeding at recommended doses."),
    ("Ibuprofen", "avoid", "Generally avoided, especially in the third trimester (risk of premature closure of the fetal ductus arteriosus).",
     "caution", "Occasional short-term use is often considered acceptable — confirm with a provider."),
    ("Aspirin", "avoid", "Regular or high-dose use is generally avoided in pregnancy; low-dose aspirin is sometimes prescribed under direct medical supervision.",
     "caution", "Regular use is generally avoided due to a theoretical Reye's-syndrome-related risk in the infant."),
    ("Warfarin", "avoid", "Crosses the placenta and is associated with birth defects, especially in the first trimester — usually substituted with a different anticoagulant during pregnancy.",
     "safe", "Considered compatible with breastfeeding — minimal transfer into breast milk."),
    ("Metformin", "caution", "Sometimes used under specialist supervision for gestational diabetes, but requires medical guidance.",
     "caution", "Generally considered low-risk but should be discussed with a provider."),
    ("Amoxicillin", "safe", "One of the more commonly used antibiotics in pregnancy when an antibiotic is needed.",
     "safe", "Considered compatible with breastfeeding."),
    ("Omeprazole", "caution", "Available data generally suggest low risk, but use should be confirmed with a provider.",
     "caution", "Limited data — discuss with a provider before regular use."),
    ("Sertraline", "caution", "May be used when the benefit outweighs the risk, under specialist supervision; some neonatal effects have been reported with third-trimester use.",
     "caution", "Often considered one of the preferred antidepressants in breastfeeding, but still requires medical guidance."),
    ("Simvastatin", "avoid", "Statins are generally avoided in pregnancy.",
     "avoid", "Generally avoided while breastfeeding due to limited safety data."),
    ("Lisinopril", "avoid", "ACE inhibitors are contraindicated, particularly in the second and third trimesters, due to risk of fetal kidney damage.",
     "caution", "Limited data — an alternative is often preferred while breastfeeding."),
]

SEED_ORGAN_IMPAIRMENT_RULES = [
    ("Metformin", "renal", "avoid",
     "Contraindicated or requires dose reduction in significant renal impairment due to increased risk of lactic acidosis."),
    ("Ibuprofen", "renal", "avoid",
     "NSAIDs reduce blood flow to the kidneys and can worsen renal impairment or cause acute kidney injury."),
    ("Ibuprofen", "hepatic", "caution",
     "Use with caution in liver impairment; may increase bleeding risk and fluid retention."),
    ("Aspirin", "renal", "caution",
     "Use with caution in renal impairment — can further reduce kidney function at higher doses."),
    ("Warfarin", "hepatic", "dose_adjust",
     "The liver produces clotting factors that Warfarin acts on — impaired liver function can unpredictably affect INR and usually requires dose adjustment and closer monitoring."),
    ("Simvastatin", "hepatic", "avoid",
     "Avoid in active liver disease or unexplained persistent elevated liver enzymes."),
    ("Lisinopril", "renal", "dose_adjust",
     "Renally cleared — dose adjustment and monitoring of kidney function and potassium are typically needed in renal impairment."),
    ("Sertraline", "hepatic", "dose_adjust",
     "Metabolized by the liver — a lower dose or slower titration is often used in hepatic impairment."),
    ("Omeprazole", "hepatic", "dose_adjust",
     "Dose adjustment may be needed in severe hepatic impairment."),
    ("Amoxicillin", "renal", "dose_adjust",
     "Renally excreted — dose or dosing interval is typically adjusted in severe renal impairment."),
]

SEED_DOSAGE_LIMITS = [
    ("Paracetamol", 500, 4000,
     "Classic OTC label maximum is 4000mg/day for adults; some current clinical guidance recommends "
     "a more conservative ceiling (e.g. 3000mg/day) for routine everyday use — check local guidance."),
    ("Ibuprofen", 200, 1200,
     "1200mg/day is the typical over-the-counter maximum; higher prescription doses exist but should "
     "only be used under medical supervision."),
    ("Aspirin", 325, 4000,
     "4000mg/day applies to pain/fever use. Low-dose 'cardioprotective' aspirin (75-100mg/day) is a "
     "completely different regimen prescribed for a different purpose."),
    ("Warfarin", 5, None,
     "Warfarin dosing is individualized and titrated to INR blood test results — there is no fixed "
     "safe daily ceiling to check against; never adjust dose without medical guidance."),
    ("Metformin", 500, 2550, "2550mg/day is a common adult maximum, often split across doses with meals."),
    ("Amoxicillin", 500, 1500, "1500mg/day is typical for standard adult infections; some regimens use higher doses."),
    ("Omeprazole", 20, 40, "40mg/day is the typical over-the-counter maximum."),
    ("Sertraline", 50, 200, "200mg/day is the typical maximum adult dose."),
    ("Simvastatin", 20, 40, "40mg/day is the usual maximum; 80mg/day is now restricted due to myopathy risk."),
    ("Lisinopril", 10, 40, "40mg/day is a common maximum adult dose for hypertension."),
]


def _seed_if_empty(conn: sqlite3.Connection):
    cur = conn.execute("SELECT COUNT(*) FROM medicines")
    if cur.fetchone()[0] > 0:
        return  # already seeded

    conn.executemany(
        "INSERT INTO medicines (name, generic_name, category, on_who_eml) VALUES (?, ?, ?, ?)",
        SEED_MEDICINES,
    )
    conn.executemany(
        "INSERT INTO drug_interactions (drug1, drug2, severity, description) "
        "VALUES (?, ?, ?, ?)",
        SEED_INTERACTIONS,
    )
    conn.executemany(
        "INSERT INTO safety_warnings (medicine, warning_type, description) "
        "VALUES (?, ?, ?)",
        SEED_WARNINGS,
    )
    conn.commit()
    logger.info("Database seeded with demo medication data.")


def _seed_dosing_safety_if_empty(conn: sqlite3.Connection):
    cur = conn.execute("SELECT COUNT(*) FROM age_dosing_rules")
    if cur.fetchone()[0] > 0:
        return  # already seeded

    conn.executemany(
        "INSERT INTO age_dosing_rules (medicine, min_age, max_age, rule_type, description) "
        "VALUES (?, ?, ?, ?, ?)",
        SEED_AGE_DOSING_RULES,
    )
    conn.executemany(
        "INSERT INTO pregnancy_rules (medicine, pregnancy_category, pregnancy_note, "
        "breastfeeding_category, breastfeeding_note) VALUES (?, ?, ?, ?, ?)",
        SEED_PREGNANCY_RULES,
    )
    conn.executemany(
        "INSERT INTO organ_impairment_rules (medicine, organ, rule_type, description) "
        "VALUES (?, ?, ?, ?)",
        SEED_ORGAN_IMPAIRMENT_RULES,
    )
    conn.executemany(
        "INSERT INTO medicine_dosage_limits (medicine, unit_dose_mg, max_daily_mg, notes) "
        "VALUES (?, ?, ?, ?)",
        SEED_DOSAGE_LIMITS,
    )
    conn.commit()
    logger.info("Database seeded with Tier 1 dosing-safety rule data.")


def init_db():
    os.makedirs(os.path.dirname(settings.DATABASE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.executescript(SCHEMA)
    _migrate_add_who_eml_column(conn)
    _seed_if_empty(conn)
    _seed_dosing_safety_if_empty(conn)
    conn.close()


def _migrate_add_who_eml_column(conn: sqlite3.Connection):
    """Idempotent migration for DBs created before the on_who_eml column
    existed (SQLite has no `ADD COLUMN IF NOT EXISTS`)."""
    try:
        conn.execute("ALTER TABLE medicines ADD COLUMN on_who_eml INTEGER DEFAULT 0")
        conn.commit()
        logger.info("Migrated: added on_who_eml column to medicines table.")
    except sqlite3.OperationalError:
        pass  # column already exists


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def find_medicine(name: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM medicines WHERE LOWER(name) = LOWER(?) "
            "OR LOWER(generic_name) = LOWER(?)",
            (name, name),
        ).fetchone()
        return row


def find_warnings(name: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM safety_warnings WHERE LOWER(medicine) = LOWER(?)",
            (name,),
        ).fetchall()
        return rows


def find_interaction(drug_a: str, drug_b: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM drug_interactions
            WHERE (LOWER(drug1) = LOWER(?) AND LOWER(drug2) = LOWER(?))
               OR (LOWER(drug1) = LOWER(?) AND LOWER(drug2) = LOWER(?))
            """,
            (drug_a, drug_b, drug_b, drug_a),
        ).fetchone()
        return row


def list_all_medicine_names() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT name FROM medicines").fetchall()
        return [r["name"] for r in rows]


# ---------------------------------------------------------------------
# Tier 1 rule-based safety lookups (see app/agents/dosing_safety_agent.py).
# ---------------------------------------------------------------------

def get_age_dosing_rules(medicine: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM age_dosing_rules WHERE LOWER(medicine) = LOWER(?)", (medicine,)
        ).fetchall()


def get_pregnancy_rule(medicine: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM pregnancy_rules WHERE LOWER(medicine) = LOWER(?)", (medicine,)
        ).fetchone()


def get_organ_impairment_rules(medicine: str, organ: Optional[str] = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if organ:
            return conn.execute(
                "SELECT * FROM organ_impairment_rules WHERE LOWER(medicine) = LOWER(?) AND organ = ?",
                (medicine, organ),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM organ_impairment_rules WHERE LOWER(medicine) = LOWER(?)", (medicine,)
        ).fetchall()


def get_dosage_limits(medicine: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM medicine_dosage_limits WHERE LOWER(medicine) = LOWER(?)", (medicine,)
        ).fetchone()


# ---------------------------------------------------------------------
# Phase 5: patient profile persistence (session-scoped, not tied to a
# real user account — just a lightweight demo of state across turns).
# ---------------------------------------------------------------------

def get_patient_profile(session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patient_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["allergies"] = [a.strip() for a in (d["allergies"] or "").split(",") if a.strip()]
        d["conditions"] = [c.strip() for c in (d["conditions"] or "").split(",") if c.strip()]
        d["current_medications"] = [m.strip() for m in (d["current_medications"] or "").split(",") if m.strip()]
        return d


def upsert_patient_profile(session_id: str, age: Optional[int] = None,
                            allergies: Optional[list[str]] = None,
                            conditions: Optional[list[str]] = None,
                            current_medications: Optional[list[str]] = None) -> dict:
    existing = get_patient_profile(session_id) or {}

    final_age = age if age is not None else existing.get("age")
    final_allergies = allergies if allergies is not None else existing.get("allergies", [])
    final_conditions = conditions if conditions is not None else existing.get("conditions", [])
    final_meds = current_medications if current_medications is not None else existing.get("current_medications", [])

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO patient_profiles (session_id, age, allergies, conditions, current_medications, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                age = excluded.age,
                allergies = excluded.allergies,
                conditions = excluded.conditions,
                current_medications = excluded.current_medications,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, final_age, ",".join(final_allergies), ",".join(final_conditions), ",".join(final_meds)),
        )
        conn.commit()

    return get_patient_profile(session_id)


# ---------------------------------------------------------------------
# Phase 4: session-scoped findings from the Vision Agent (e.g. a chest
# X-ray analysis) or the new OCR Agent (a photographed prescription/
# label), so a later text query in the same session can be synthesized
# together with the earlier finding by the Coordinator.
# ---------------------------------------------------------------------

def save_finding(session_id: str, finding_text: str, source: str = "vision_agent") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO session_findings (session_id, finding_text, source) VALUES (?, ?, ?)",
            (session_id, finding_text, source),
        )
        conn.commit()


def get_latest_finding(session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM session_findings WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------
# New: multi-turn conversational memory (see app/agents/conversation_memory.py).
# Session-scoped, most-recent-first storage of user/assistant turns.
# ---------------------------------------------------------------------

def save_conversation_turn(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversation_turns (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def get_recent_conversation_turns(session_id: str, limit: int = 4) -> list[dict]:
    """Returns the most recent `limit` turns (user+assistant messages combined,
    not pairs) in chronological order (oldest of the window first)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM conversation_turns "
            "WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
            (session_id, limit * 2),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------
# New: full reasoning-trace export (see app/trace/reasoning_trace.py).
# Stores the entire agent-by-agent trace per query as a JSON blob,
# retrievable via GET /api/trace/{query_id}.
# ---------------------------------------------------------------------

def save_trace(query_id: str, session_id: Optional[str], query_text: str, trace_json: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reasoning_traces (query_id, session_id, query_text, trace_json) "
            "VALUES (?, ?, ?, ?)",
            (query_id, session_id, query_text, trace_json),
        )
        conn.commit()


def get_trace(query_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM reasoning_traces WHERE query_id = ?", (query_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------
# Novelty addition: lightweight feedback loop ("continual improvement
# without full retraining"). Raw feedback is stored for later manual
# review/curation, and — where the feedback references specific RAG
# document ids — a per-document trust multiplier is nudged up/down and
# applied as a re-ranking factor at retrieval time (see
# app/rag/chroma_store.py and app/rag/knowledge_base.py). This is a
# real, working mechanism, but it is intentionally simple: it reweights
# existing documents, it does not learn new facts or retrain any model.
# ---------------------------------------------------------------------

TRUST_STEP_DOWN = 0.85   # multiplicative penalty per "not_helpful" vote
TRUST_STEP_UP = 1.05     # multiplicative reward per "helpful" vote (capped at 1.0)
TRUST_FLOOR = 0.25       # a doc can be down-weighted but never fully silenced


def save_feedback(query: str, rating: str, query_type: Optional[str] = None,
                   session_id: Optional[str] = None, comment: Optional[str] = None,
                   source_ref_ids: Optional[list[str]] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (session_id, query, query_type, rating, comment) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, query, query_type, rating, comment),
        )
        conn.commit()

    for doc_id in (source_ref_ids or []):
        _adjust_trust(doc_id, positive=(rating == "helpful"))


def _adjust_trust(doc_id: str, positive: bool) -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM doc_trust_scores WHERE doc_id = ?", (doc_id,)).fetchone()
        current = row["trust_multiplier"] if row else 1.0
        pos = row["positive_count"] if row else 0
        neg = row["negative_count"] if row else 0

        if positive:
            new_trust = min(1.0, current * TRUST_STEP_UP)
            pos += 1
        else:
            new_trust = max(TRUST_FLOOR, current * TRUST_STEP_DOWN)
            neg += 1

        conn.execute(
            """
            INSERT INTO doc_trust_scores (doc_id, trust_multiplier, positive_count, negative_count, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(doc_id) DO UPDATE SET
                trust_multiplier = excluded.trust_multiplier,
                positive_count = excluded.positive_count,
                negative_count = excluded.negative_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (doc_id, new_trust, pos, neg),
        )
        conn.commit()


def get_trust_multiplier(doc_id: str) -> float:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT trust_multiplier FROM doc_trust_scores WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row["trust_multiplier"] if row else 1.0


# ---------------------------------------------------------------------
# New: authentication (users + sessions). See app/auth.py for the
# hashing/token logic — this module only persists rows.
# ---------------------------------------------------------------------

def create_user(name: str, email: str, password_hash: str, salt: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, salt),
        )
        conn.commit()
        return cur.lastrowid


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email,)).fetchone()


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_session(token: str, user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
        conn.commit()


def get_user_by_session_token(token: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT u.id, u.name, u.email, u.created_at FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# ---------------------------------------------------------------------
# New: history & saved results (powers the Dashboard's "Recent Activity"
# / "Saved Results" panels and the dedicated History & Saved Results page).
# ---------------------------------------------------------------------

def add_history_item(session_id: str, item_type: str, title: str, preview: str,
                      confidence: Optional[float] = None, verification_status: Optional[str] = None,
                      payload: Optional[dict] = None) -> int:
    import json as _json
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO history_items (session_id, item_type, title, preview, confidence, "
            "verification_status, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, item_type, title, preview[:400], confidence, verification_status,
             _json.dumps(payload) if payload is not None else None),
        )
        conn.commit()
        return cur.lastrowid


def list_history(session_id: str, item_type: Optional[str] = None, saved_only: bool = False,
                  limit: int = 50, offset: int = 0) -> list[dict]:
    query = "SELECT * FROM history_items WHERE session_id = ?"
    params: list = [session_id]
    if item_type and item_type != "all":
        query += " AND item_type = ?"
        params.append(item_type)
    if saved_only:
        query += " AND saved = 1"
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def count_history_by_type(session_id: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT item_type, COUNT(*) c FROM history_items WHERE session_id = ? GROUP BY item_type",
            (session_id,),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) c FROM history_items WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
    counts = {r["item_type"]: r["c"] for r in rows}
    counts["all"] = total
    return counts


def toggle_history_saved(item_id: int, session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM history_items WHERE id = ? AND session_id = ?", (item_id, session_id)
        ).fetchone()
        if not row:
            return None
        new_saved = 0 if row["saved"] else 1
        conn.execute("UPDATE history_items SET saved = ? WHERE id = ?", (new_saved, item_id))
        conn.commit()
        return {**dict(row), "saved": new_saved}


def delete_history_item(item_id: int, session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM history_items WHERE id = ? AND session_id = ?", (item_id, session_id))
        conn.commit()
        return cur.rowcount > 0


def get_feedback_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
        helpful = conn.execute("SELECT COUNT(*) c FROM feedback WHERE rating='helpful'").fetchone()["c"]
        not_helpful = conn.execute("SELECT COUNT(*) c FROM feedback WHERE rating='not_helpful'").fetchone()["c"]
        rows = conn.execute(
            "SELECT query_type, rating, COUNT(*) c FROM feedback "
            "WHERE query_type IS NOT NULL GROUP BY query_type, rating"
        ).fetchall()

    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        by_type.setdefault(r["query_type"], {"helpful": 0, "not_helpful": 0})
        by_type[r["query_type"]][r["rating"]] = r["c"]

    return {"total": total, "helpful": helpful, "not_helpful": not_helpful, "by_query_type": by_type}
