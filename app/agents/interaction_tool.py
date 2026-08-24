"""
3.4 Drug Interaction Checking Tool

Receives medicine names, normalizes them against the known medicine list,
searches the drug interaction database (via the SQL Agent), and returns a
structured interaction result to the Coordinator / Pharmacist Agent.
"""
import logging
from dataclasses import dataclass
from difflib import get_close_matches

from app.agents.sql_agent import sql_agent
from app.database import list_all_medicine_names

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

        result = sql_agent.lookup_interaction(norm_a, norm_b)
        if not result.found:
            return InteractionCheckResult(
                drug_a=drug_a, drug_b=drug_b,
                normalized_a=norm_a, normalized_b=norm_b,
                status="no_known_interaction",
            )

        interaction = result.interaction
        return InteractionCheckResult(
            drug_a=drug_a, drug_b=drug_b,
            normalized_a=norm_a, normalized_b=norm_b,
            status="found",
            severity=interaction["severity"],
            description=interaction["description"],
        )


drug_interaction_tool = DrugInteractionTool()
