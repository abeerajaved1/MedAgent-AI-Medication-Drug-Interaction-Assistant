"""
Emergency / Triage Agent (new capability — named in the original blueprint,
previously 0% built).

Runs BEFORE the normal RAG / SQL / Pharmacist / Safety pipeline on every
incoming query. If the query contains language consistent with a medical
or psychiatric emergency, the pipeline short-circuits and returns an
immediate "seek emergency care" response instead of proceeding through
retrieval and drafting — matching real-world triage behavior (you don't
look up a drug interaction while someone is describing chest pain).

Detection is two-layered, cheap, and fails safe:
  1. Deterministic keyword/phrase matching (fast, zero cost, works even
     in LLM mock mode, cannot be "argued out of" by a jailbreak attempt).
  2. An LLM classification pass as a second opinion for phrasing the
     keyword list doesn't catch — but the keyword layer alone is already
     sufficient to trigger, since false positives here are far cheaper
     than false negatives.

This agent never diagnoses and never says what the emergency IS beyond
the category it matched — it only classifies urgency and routes to
emergency guidance.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from app.llm_client import llm_client

logger = logging.getLogger("medagent.emergency_agent")

# Category -> list of regex-safe phrases/keywords. Kept broad and plain-
# language on purpose (this is what real people type, not clinical terms).
RED_FLAG_PATTERNS: dict[str, list[str]] = {
    "cardiac_respiratory": [
        r"chest pain", r"tight(ness)? in (my|the) chest", r"crushing pain",
        r"can'?t breathe", r"cannot breathe", r"difficulty breathing",
        r"shortness of breath", r"gasping for air", r"turning blue",
        r"lips? (are |is )?blue", r"choking",
    ],
    "severe_bleeding_trauma": [
        r"severe bleeding", r"won'?t stop bleeding", r"bleeding a lot",
        r"heavy bleeding", r"deep cut", r"stabbed", r"gunshot",
        r"major (car )?accident", r"lost consciousness", r"passed out",
        r"unresponsive", r"not breathing",
    ],
    "neurological": [
        r"stroke", r"face (is )?droop", r"slurred speech", r"sudden numbness",
        r"sudden weakness on one side", r"worst headache of (my|her|his) life",
        r"seizure", r"convuls",
    ],
    "allergic_anaphylaxis": [
        r"anaphyla", r"throat (is )?closing", r"swelling of (the |my )?(face|throat|tongue)",
        r"can'?t swallow", r"severe allergic reaction", r"hives? and (trouble|difficulty) breathing",
    ],
    "overdose_poisoning": [
        r"overdos", r"took too many pills", r"swallowed (a )?(bottle|poison|bleach|chemical)",
        r"accidental(ly)? poison",
    ],
    "suicidal_self_harm": [
        r"kill myself", r"suicid", r"end my life", r"want to die",
        r"hurt myself", r"self.?harm", r"no reason to live",
        r"better off dead", r"planning to (die|end it)",
    ],
}

CLASSIFY_SYSTEM_PROMPT = """You are an Emergency Triage classifier for a medication-information
assistant. You will be given a single user message. Decide ONLY whether it describes
a potential medical or psychiatric EMERGENCY requiring immediate real-world action
(calling emergency services, going to an ER, or an immediate crisis line) — as opposed
to a routine question about medicines, side effects, or general health information.

Return STRICT JSON only, in this exact shape:
{"is_emergency": true | false,
 "category": "cardiac_respiratory" | "severe_bleeding_trauma" | "neurological" |
              "allergic_anaphylaxis" | "overdose_poisoning" | "suicidal_self_harm" | "none",
 "reasoning": "one short sentence"}

Err on the side of true if there is real ambiguity involving potential severe harm.
Do NOT classify routine medication questions (dosage, side effects, interactions,
"is it safe to take X") as emergencies unless they also describe acute symptoms.
"""

EMERGENCY_MESSAGES: dict[str, str] = {
    "cardiac_respiratory": (
        "This may describe a medical emergency (possible cardiac or breathing "
        "problem). Please call your local emergency number (e.g. 911 / 999 / 112) "
        "or go to the nearest emergency room right now. Do not wait for an app to "
        "help you decide — this needs immediate in-person medical attention."
    ),
    "severe_bleeding_trauma": (
        "This may describe a medical emergency (severe bleeding, trauma, or loss "
        "of consciousness). Please call your local emergency number immediately "
        "or get to the nearest emergency room. Apply firm pressure to any bleeding "
        "wound while help is on the way if you are able to."
    ),
    "neurological": (
        "This may describe signs of a stroke or a neurological emergency (facial "
        "drooping, slurred speech, sudden weakness/numbness, seizure, or the "
        "'worst headache of your life'). Please call your local emergency number "
        "immediately — treatment within the first hours matters a great deal."
    ),
    "allergic_anaphylaxis": (
        "This may describe a severe allergic reaction (anaphylaxis). If you have "
        "an epinephrine auto-injector, use it now and call your local emergency "
        "number immediately, even if symptoms seem to improve afterward."
    ),
    "overdose_poisoning": (
        "This may describe a poisoning or overdose emergency. Please call your "
        "local emergency number or a poison control center immediately. If "
        "possible, keep the medication/substance container to show responders."
    ),
    "suicidal_self_harm": (
        "It sounds like you might be going through something really difficult "
        "right now, and I want to make sure you get real support. If you are in "
        "immediate danger, please call your local emergency number now. You can "
        "also reach a crisis line: in the US, call or text 988 (Suicide & Crisis "
        "Lifeline); in the UK/ROI, Samaritans at 116 123; or search 'suicide "
        "helpline' plus your country for a local option. You don't have to go "
        "through this alone."
    ),
}


@dataclass
class EmergencyCheckResult:
    is_emergency: bool
    category: str = "none"
    matched_flags: list[str] = field(default_factory=list)
    message: str = ""
    detection_method: str = "none"   # "keyword" | "llm" | "none"


class EmergencyAgent:
    def check(self, user_query: str) -> EmergencyCheckResult:
        query_lower = user_query.lower()

        # --- Layer 1: deterministic keyword matching (cannot be bypassed) ---
        for category, patterns in RED_FLAG_PATTERNS.items():
            matched = [p for p in patterns if re.search(p, query_lower)]
            if matched:
                logger.warning(f"Emergency Agent: keyword match, category={category}, patterns={matched}")
                return EmergencyCheckResult(
                    is_emergency=True,
                    category=category,
                    matched_flags=matched,
                    message=EMERGENCY_MESSAGES[category],
                    detection_method="keyword",
                )

        # --- Layer 2: LLM second-opinion classification (skipped in mock mode —
        # the keyword layer already covers the demo/offline case) ---
        if not llm_client.mock_mode:
            try:
                raw = llm_client.complete(
                    CLASSIFY_SYSTEM_PROMPT, f"User message: {user_query}",
                    json_mode=True, temperature=0.0,
                )
                data = json.loads(raw)
                if data.get("is_emergency") and data.get("category") in EMERGENCY_MESSAGES:
                    category = data["category"]
                    logger.warning(f"Emergency Agent: LLM classified as emergency, category={category}")
                    return EmergencyCheckResult(
                        is_emergency=True,
                        category=category,
                        matched_flags=[data.get("reasoning", "")],
                        message=EMERGENCY_MESSAGES[category],
                        detection_method="llm",
                    )
            except Exception as e:
                logger.info(f"Emergency Agent LLM classification skipped/failed: {e}")

        return EmergencyCheckResult(is_emergency=False)


emergency_agent = EmergencyAgent()
