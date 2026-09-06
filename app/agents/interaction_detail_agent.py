"""
Interaction Detail Agent (new — supports the Drug Interaction Checker
UI page's structured layout: risk level, evidence-strength stars,
onset/severity/frequency/reliability fields, "why is this a concern",
possible effects list, and "what you should do" guidance).

Builds on the existing DrugInteractionTool (app/agents/interaction_tool.py)
and ExternalSourceAgent for evidence, then asks the LLM to write the
explanatory fields — grounded in that evidence only, same discipline as
the Pharmacist Agent.
"""
import json
import logging
from dataclasses import dataclass, field

from app.llm_client import llm_client
from app.agents.interaction_tool import drug_interaction_tool
from app.agents.external_source_agent import external_source_agent

logger = logging.getLogger("medagent.interaction_detail_agent")

EXPLAIN_SYSTEM_PROMPT = """You are explaining a drug-drug interaction for a structured UI panel. Use
ONLY the evidence given. Return STRICT JSON only, in this exact shape:
{"why_concern": "two or three plain-language sentences explaining the mechanism/reason",
 "possible_effects": ["short phrase", "..."],
 "what_you_should_do": ["short actionable sentence", "..."],
 "seek_care_if": "one short sentence describing when to seek urgent care, or null if not applicable"}
Keep possible_effects and what_you_should_do items short (under ~14 words each). Do not invent a
mechanism that isn't supported by the evidence — if the evidence is thin, say so plainly instead of
elaborating.
"""

# Deterministic mappings from severity -> the UI's other structured fields.
# These are heuristic presentation choices, not clinical claims beyond what
# the severity label itself already asserts (see interaction_tool.py).
SEVERITY_TO_RISK_LEVEL = {"major": "HIGH", "moderate": "MODERATE", "minor": "LOW"}
SEVERITY_TO_STARS = {"major": 5, "moderate": 3, "minor": 2}


@dataclass
class InteractionDetail:
    status: str                      # found | no_known_interaction | not_found
    drug_a: str
    drug_b: str
    severity: str | None = None
    risk_level: str | None = None
    evidence_stars: int | None = None
    why_concern: str | None = None
    possible_effects: list[str] = field(default_factory=list)
    what_you_should_do: list[str] = field(default_factory=list)
    seek_care_if: str | None = None
    sources: list[dict] = field(default_factory=list)


class InteractionDetailAgent:
    def analyze(self, drug_a: str, drug_b: str) -> InteractionDetail:
        check = drug_interaction_tool.check(drug_a, drug_b)

        if check.status == "unknown_medicine":
            return InteractionDetail(status="not_found", drug_a=check.normalized_a, drug_b=check.normalized_b)

        sources = []
        evidence_parts = []

        if check.status == "found":
            evidence_parts.append(
                f"Local interaction database: {check.normalized_a} + {check.normalized_b} "
                f"(severity: {check.severity}) — {check.description}"
            )
            sources.append({"source": "MedAgent interaction database", "snippet": check.description[:200],
                             "type": "Drug Database", "relevance": "Very High"})

        ext_hint = external_source_agent.check_interaction_hint(check.normalized_a, check.normalized_b)
        if ext_hint:
            evidence_parts.append(f"{ext_hint.source_name}: {ext_hint.text}")
            sources.append({"source": ext_hint.source_name, "snippet": ext_hint.text[:200],
                             "type": "External Reference", "relevance": "High"})

        if check.status == "no_known_interaction":
            return InteractionDetail(
                status="no_known_interaction", drug_a=check.normalized_a, drug_b=check.normalized_b,
                severity="none", risk_level="LOW", evidence_stars=1,
                why_concern="No known interaction was found between these medicines in the available data.",
                sources=sources,
            )

        explained = self._explain_with_llm(check.normalized_a, check.normalized_b,
                                            check.severity, "\n\n".join(evidence_parts))

        return InteractionDetail(
            status="found", drug_a=check.normalized_a, drug_b=check.normalized_b,
            severity=check.severity,
            risk_level=SEVERITY_TO_RISK_LEVEL.get(check.severity, "MODERATE"),
            evidence_stars=SEVERITY_TO_STARS.get(check.severity, 3),
            why_concern=explained.get("why_concern", check.description),
            possible_effects=explained.get("possible_effects", []),
            what_you_should_do=explained.get("what_you_should_do", [
                "Do not combine these medicines without medical advice.",
                "Contact a healthcare professional before combining them.",
            ]),
            seek_care_if=explained.get("seek_care_if"),
            sources=sources,
        )

    def _explain_with_llm(self, drug_a: str, drug_b: str, severity: str, evidence_text: str) -> dict:
        if llm_client.mock_mode:
            return {
                "why_concern": f"[MOCK MODE] {drug_a} and {drug_b} have a known {severity} interaction "
                               f"per the local database — configure an LLM API key for a full explanation.",
                "possible_effects": [], "what_you_should_do": [], "seek_care_if": None,
            }
        try:
            raw = llm_client.complete(
                EXPLAIN_SYSTEM_PROMPT,
                f"Drug pair: {drug_a} + {drug_b}\nSeverity: {severity}\n\n"
                f"--- Evidence ---\n{evidence_text}\n--- End evidence ---",
                json_mode=True, temperature=0.2,
            )
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Interaction Detail Agent explanation failed for {drug_a}+{drug_b}: {e}")
            return {}


interaction_detail_agent = InteractionDetailAgent()
