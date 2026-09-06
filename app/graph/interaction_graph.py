"""
Phase 3 upgrade: drug-interaction data as a graph (NetworkX) instead of a
flat SQL table lookup, with an optional loader for the free, public
TWOSIDES dataset so real-world interaction coverage can go far beyond the
~7 hand-seeded rows in the demo database.

TWOSIDES (Tatonetti Lab, Stanford) is a free, publicly downloadable
research dataset of drug-drug-interaction/adverse-event associations
derived from real-world data (no signup / no cost):
  https://tatonettilab.org/offsides/
  https://github.com/tatonetti-lab/twosides

TWOSIDES gives statistical association scores (e.g. PRR — proportional
reporting ratio) between drug pairs and adverse events, NOT a
clinician-assigned severity label. This loader maps PRR to a severity
bucket using a documented, adjustable heuristic — this is explicitly a
heuristic, not clinical-grade severity grading, and should be described
as such in any write-up.

By default the graph is built from the local SQL seed data (small,
curated, has real severity labels). Call `load_twosides_csv(...)` to
merge in a downloaded TWOSIDES sample for broader coverage.
"""
import csv
import logging
from dataclasses import dataclass

import networkx as nx

from app import database as db

logger = logging.getLogger("medagent.interaction_graph")

SEVERITY_RANK = {"minor": 1, "moderate": 2, "major": 3}

# PRR -> severity heuristic bucket (adjust based on your own analysis /
# literature review before using this for anything beyond a demo).
PRR_SEVERITY_THRESHOLDS = [
    (10.0, "major"),
    (3.0, "moderate"),
    (0.0, "minor"),
]


@dataclass
class InteractionEdge:
    severity: str
    description: str
    source: str  # "local_db" | "twosides"


class InteractionGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._loaded = False  # lazy: DB may not be initialized yet at import time

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            with db.get_conn() as conn:
                rows = conn.execute("SELECT * FROM drug_interactions").fetchall()
        except Exception as e:
            logger.warning(f"Interaction graph could not load from DB yet ({e}); will retry on next access.")
            return
        for row in rows:
            self.add_edge(row["drug1"], row["drug2"], row["severity"],
                          row["description"], source="local_db")
        self._loaded = True
        logger.info(f"Interaction graph loaded from local DB: "
                    f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def add_edge(self, drug_a: str, drug_b: str, severity: str, description: str, source: str = "local_db"):
        a, b = drug_a.strip(), drug_b.strip()
        if self.graph.has_edge(a, b):
            existing: InteractionEdge = self.graph[a][b]["data"]
            # Keep the higher-severity, more informative edge if we see the pair twice.
            if SEVERITY_RANK.get(severity, 0) <= SEVERITY_RANK.get(existing.severity, 0):
                return
        self.graph.add_edge(a, b, data=InteractionEdge(severity=severity, description=description, source=source))

    def load_twosides_csv(self, csv_path: str,
                           drug1_col: str = "drug_1_concept_name",
                           drug2_col: str = "drug_2_concept_name",
                           event_col: str = "condition_meddra_name",
                           prr_col: str = "PRR",
                           max_rows: int | None = None):
        """
        Optional: merge a downloaded TWOSIDES CSV sample into the graph for
        broader interaction coverage. Column names above match the common
        TWOSIDES export format — adjust if your download differs.
        """
        self._ensure_loaded()
        added = 0
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break
                    try:
                        prr = float(row.get(prr_col, 0) or 0)
                    except ValueError:
                        prr = 0.0
                    severity = "minor"
                    for threshold, label in PRR_SEVERITY_THRESHOLDS:
                        if prr >= threshold:
                            severity = label
                            break
                    event = row.get(event_col, "an adverse event")
                    description = (
                        f"TWOSIDES data associates this combination with increased "
                        f"reports of {event} (PRR={prr:.2f})."
                    )
                    self.add_edge(row[drug1_col], row[drug2_col], severity, description, source="twosides")
                    added += 1
        except FileNotFoundError:
            logger.warning(f"TWOSIDES CSV not found at {csv_path} — skipping import.")
            return 0
        logger.info(f"Merged {added} TWOSIDES interaction rows into the graph.")
        return added

    def has_interaction(self, drug_a: str, drug_b: str) -> bool:
        self._ensure_loaded()
        return self.graph.has_edge(drug_a, drug_b)

    def get_interaction(self, drug_a: str, drug_b: str) -> InteractionEdge | None:
        self._ensure_loaded()
        if not self.graph.has_edge(drug_a, drug_b):
            return None
        return self.graph[drug_a][drug_b]["data"]

    def neighbors(self, drug: str) -> list[str]:
        """All drugs with a known direct interaction with `drug` — useful for
        polypharmacy checks against a patient's existing medication list."""
        self._ensure_loaded()
        if drug not in self.graph:
            return []
        return list(self.graph.neighbors(drug))

    def check_against_medication_list(self, new_drug: str, current_meds: list[str]) -> list[tuple[str, InteractionEdge]]:
        """Given a drug the patient is considering, check it against every
        drug already on their profile's medication list. Returns all hits."""
        self._ensure_loaded()
        hits = []
        for med in current_meds:
            edge = self.get_interaction(new_drug, med)
            if edge:
                hits.append((med, edge))
        return hits


interaction_graph = InteractionGraph()
