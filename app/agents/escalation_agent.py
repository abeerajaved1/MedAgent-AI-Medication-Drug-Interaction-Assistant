"""
Escalate-to-human trigger (new capability — formalizes the blueprint's
Phase 5 "uncertainty quantification and escalation triggers" requirement
into an explicit, structured, citable safety mechanism instead of an
implicit side-effect of verification_status/confidence_score).

Produces a single boolean + a short human-readable reason, so a caller
(or a UI) can act on "this needs a human" as a first-class signal rather
than inferring it from several other fields.

Deterministic and rule-based on purpose — this must be an auditable
safety gate, not something the LLM could be argued out of.
"""
import logging
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger("medagent.escalation_agent")

CONFIDENCE_ESCALATION_THRESHOLD = settings.ESCALATION_CONFIDENCE_THRESHOLD


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str | None = None


class EscalationAgent:
    def evaluate(self, verification_status: str, confidence_score: float | None,
                 severity: str | None, revision_count: int, max_veto_rounds: int,
                 self_consistency_flagged: bool = False,
                 profile_has_concerns: bool = False,
                 emergency_triggered: bool = False) -> EscalationDecision:

        if emergency_triggered:
            return EscalationDecision(
                escalate=True,
                reason="Emergency Agent detected red-flag language — this always requires "
                       "immediate real-world human/professional attention.",
            )

        if verification_status == "insufficient_evidence":
            return EscalationDecision(
                escalate=True,
                reason="No reliable evidence was available to answer this query confidently.",
            )

        if severity == "major":
            return EscalationDecision(
                escalate=True,
                reason="A major drug interaction or safety concern was identified.",
            )

        if profile_has_concerns:
            return EscalationDecision(
                escalate=True,
                reason="The patient profile check found an allergy or polypharmacy concern "
                       "that should be reviewed by a pharmacist or doctor.",
            )

        if revision_count >= max_veto_rounds and verification_status != "approved":
            return EscalationDecision(
                escalate=True,
                reason="The Safety Agent did not approve the response even after the maximum "
                       "number of revision rounds.",
            )

        if self_consistency_flagged:
            return EscalationDecision(
                escalate=True,
                reason="Two independently drafted responses to this query disagreed on "
                       "safety-relevant content, indicating higher uncertainty than usual.",
            )

        if confidence_score is not None and confidence_score < CONFIDENCE_ESCALATION_THRESHOLD:
            return EscalationDecision(
                escalate=True,
                reason=f"Overall confidence ({confidence_score:.2f}) fell below the "
                       f"escalation threshold ({CONFIDENCE_ESCALATION_THRESHOLD}).",
            )

        return EscalationDecision(escalate=False, reason=None)


escalation_agent = EscalationAgent()
