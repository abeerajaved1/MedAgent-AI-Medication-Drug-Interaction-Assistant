"""
3.4 Drug Interaction Checking Tool  (Phase 3: now graph-backed)

Receives medicine names, normalizes them against the known medicine list,
and queries the NetworkX-based interaction graph (app/graph/interaction_graph.py)
instead of a flat SQL row lookup. The graph representation also enables
polypharmacy checks: a new drug can be checked against a patient's whole
current medication list, not just a single pair.
"""
import logging
from dataclasses import dataclass
from difflib import get_close_matches

from app.database import list_all_medicine_names
from app.graph.interaction_graph import interaction_graph, InteractionEdge

logger = logging.getLogger("medagent.interaction_tool")


@dataclass
class InteractionCheckResult:
    drug_a: str
    drug_b: str
    normalized_a: str
    normalized_b: str
    status: str            # "no_known_interaction" | "found" | "unknown_medicine"
    severity: str | None = None
    description: str | None = None


@dataclass
class PolypharmacyHit:
    against_drug: str
    severity: str
    description: str


class DrugInteractionTool:
    def __init__(self):
        self._known: list[str] | None = None  # lazy-loaded (DB may not be ready yet)

    @property
    def known(self) -> list[str]:
        if self._known is None:
            self._known = list_all_medicine_names()
        return self._known

    def _normalize(self, name: str) -> str | None:
        matches = get_close_matches(name, self.known, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def check(self, drug_a: str, drug_b: str) -> InteractionCheckResult:
        norm_a = self._normalize(drug_a)
        norm_b = self._normalize(drug_b)

        if not norm_a or not norm_b:
            return InteractionCheckResult(
                drug_a=drug_a, drug_b=drug_b,
                normalized_a=norm_a or drug_a, normalized_b=norm_b or drug_b,
                status="unknown_medicine",
            )

        edge: InteractionEdge | None = interaction_graph.get_interaction(norm_a, norm_b)
        if not edge:
            return InteractionCheckResult(
                drug_a=drug_a, drug_b=drug_b,
                normalized_a=norm_a, normalized_b=norm_b,
                status="no_known_interaction",
            )

        return InteractionCheckResult(
            drug_a=drug_a, drug_b=drug_b,
            normalized_a=norm_a, normalized_b=norm_b,
            status="found",
            severity=edge.severity,
            description=edge.description,
        )

    def check_against_medication_list(self, new_drug: str, current_meds: list[str]) -> list[PolypharmacyHit]:
        """Phase 5 support: check a candidate drug against every medicine
        already on a patient's profile, using the interaction graph's
        neighbor lookup (a 1-hop polypharmacy check)."""
        norm_new = self._normalize(new_drug)
        if not norm_new:
            return []
        norm_meds = [self._normalize(m) for m in current_meds if self._normalize(m)]
        hits = interaction_graph.check_against_medication_list(norm_new, norm_meds)
        return [PolypharmacyHit(against_drug=med, severity=edge.severity, description=edge.description)
                for med, edge in hits]


drug_interaction_tool = DrugInteractionTool()
