"""
Adaptive clarification before answering (new capability — models real
pharmacist behavior: a pharmacist who is missing a safety-relevant fact
asks for it before answering, rather than guessing).

Runs after the Coordinator's routing decision (and after the Emergency
Agent, which always takes priority). Uses cheap deterministic rules
first (age-sensitive phrasing without a known age, an interaction
question naming fewer than 2 medicines, a very short/ambiguous query),
then an optional LLM pass for subtler cases. When triggered, the
pipeline should short-circuit and return the clarifying question
instead of proceeding through RAG/Pharmacist/Safety on a guess.

Kept conservative on purpose: this must not fire on every query (that
would be annoying, not helpful), only when answering blind would mean
guessing at something safety-relevant.
"""
import json
import logging
import re
from dataclasses import dataclass

from app.llm_client import llm_client

logger = logging.getLogger("medagent.clarification_agent")

AGE_SENSITIVE_PATTERNS = [
    r"\bmy (child|kid|son|daughter|baby|toddler|infant)\b",
    r"\bfor a (child|kid|baby|toddler|infant)\b",
    r"\belderly\b",
]
# NOTE: "pregnant"/"breastfeeding" were deliberately removed from the list
# above. They used to trigger an age clarification question, but the new
# Tier 1 Dosing Safety Agent (app/agents/dosing_safety_agent.py) checks
# pregnancy/breastfeeding contraindications directly from the word itself
# — no age is needed for that rule table, so asking for one here would be
# an unnecessary, wrong follow-up question that also short-circuits the
# pipeline before the dosing safety check ever runs.

VAGUE_REFERENCE_PATTERNS = [
    r"\bmy other medication\b", r"\bthe other one\b", r"\bthat (drug|medicine|pill)\b",
    r"\bwhat about with\b",
]


@dataclass
class ClarificationResult:
    needs_clarification: bool
    question: str | None = None
    reason: str | None = None


class ClarificationAgent:
    def check(self, user_query: str, decision, profile: dict | None,
              has_conversation_history: bool) -> ClarificationResult:
        query_lower = user_query.lower()

        # Rule 1: age-sensitive phrasing with no age on file and no age mentioned inline.
        if any(re.search(p, query_lower) for p in AGE_SENSITIVE_PATTERNS):
            has_age_on_file = bool(profile and profile.get("age") is not None)
            has_inline_age = bool(re.search(r"\b\d{1,3}\s*(years?|yrs?|months?)\s*old\b", query_lower))
            if not has_age_on_file and not has_inline_age:
                return ClarificationResult(
                    needs_clarification=True,
                    question=(
                        "To answer this safely, could you tell me the age (or age range, "
                        "e.g. 'infant', 'child', 'adult') of the person this is for? Dosing "
                        "and safety can differ a lot by age."
                    ),
                    reason="age_sensitive_query_missing_age",
                )

        # Rule 2: drug-interaction intent named fewer than 2 medicines, and the
        # query uses a vague reference instead of naming the second medicine —
        # and there's no conversation history to resolve it from, and no
        # current_medications on the profile to fall back on.
        if decision.intent == "drug_interaction" and len(decision.medicines) < 2:
            has_vague_reference = any(re.search(p, query_lower) for p in VAGUE_REFERENCE_PATTERNS)
            has_profile_meds = bool(profile and profile.get("current_medications"))
            if has_vague_reference and not has_conversation_history and not has_profile_meds:
                return ClarificationResult(
                    needs_clarification=True,
                    question=(
                        "Which specific medicine did you mean by 'the other one'? I want to "
                        "check the interaction against the exact medicine name."
                    ),
                    reason="vague_medicine_reference_unresolved",
                )
            if not has_vague_reference and not has_profile_meds and not has_conversation_history:
                return ClarificationResult(
                    needs_clarification=True,
                    question=(
                        "Could you name the second medicine you'd like me to check this "
                        "against? I only caught one medicine name in your message."
                    ),
                    reason="interaction_query_missing_second_medicine",
                )

        return ClarificationResult(needs_clarification=False)


clarification_agent = ClarificationAgent()
