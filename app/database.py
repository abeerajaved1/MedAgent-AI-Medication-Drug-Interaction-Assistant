"""
SQLite database layer: schema creation + seed data + query helpers
for MEDICINES, DRUG_INTERACTIONS, SAFETY_WARNINGS.
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
    category TEXT
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
"""

# NOTE: This is a small illustrative dataset for demo purposes only.
# It is NOT a complete or clinically validated drug interaction database.
SEED_MEDICINES = [
    ("Paracetamol", "Acetaminophen", "Analgesic / Antipyretic"),
    ("Ibuprofen", "Ibuprofen", "NSAID"),
    ("Aspirin", "Acetylsalicylic acid", "NSAID / Antiplatelet"),
    ("Warfarin", "Warfarin sodium", "Anticoagulant"),
    ("Metformin", "Metformin HCl", "Antidiabetic (Biguanide)"),
    ("Amoxicillin", "Amoxicillin", "Antibiotic (Penicillin)"),
    ("Omeprazole", "Omeprazole", "Proton Pump Inhibitor"),
    ("Sertraline", "Sertraline HCl", "SSRI Antidepressant"),
    ("Simvastatin", "Simvastatin", "Statin"),
    ("Lisinopril", "Lisinopril", "ACE Inhibitor"),
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


def _seed_if_empty(conn: sqlite3.Connection):
    cur = conn.execute("SELECT COUNT(*) FROM medicines")
    if cur.fetchone()[0] > 0:
        return  # already seeded

    conn.executemany(
        "INSERT INTO medicines (name, generic_name, category) VALUES (?, ?, ?)",
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


def init_db():
    os.makedirs(os.path.dirname(settings.DATABASE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.executescript(SCHEMA)
    _seed_if_empty(conn)
    conn.close()


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
