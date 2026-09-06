"""
Symptom Checker Agent (new capability — named in the original blueprint,
previously 0% built).

Takes a free-text symptom description and produces a cautious, explicitly
non-diagnostic differential: a short list of possibilities with a
confidence LABEL (not a calibrated probability), each phrased as "could
be consistent with" rather than a diagnosis. This mirrors the Vision
Agent's framing (app/agents/vision_agent.py) for the same reason: an LLM
run through a chat endpoint has no clinical authority to diagnose.

Feeds into the Pharmacist Agent as an additional evidence block so a
symptom question and a medication question in the same turn ("I have a
headache and I'm on Warfarin, can I take ibuprofen?") can be answered
together in one synthesized response.
"""
import json
import logging
from dataclasses import dataclass, field

from app.llm_client import llm_client

logger = logging.getLogger("medagent.symptom_checker")

VALID_CONFIDENCE = {"low", "medium", "higher_but_non_diagnostic"}

SYMPTOM_SYSTEM_PROMPT = """You are a Symptom Checker Agent inside a medication-information
assistant. You are NOT a doctor and must NEVER diagnose. Given a plain-language
description of symptoms (and optional patient context), produce a short, cautious
differential of possibilities.

Return STRICT JSON only, in this exact shape:
{"possibilities": [
    {"condition": "short name", "confidence": "low" | "medium" | "higher_but_non_diagnostic",
     "reasoning": "one short plain-language sentence tying it to the described symptoms"}
  ],
 "red_flags_to_watch_for": ["short phrase", "..."],
 "general_advice": "one or two calm, non-alarming sentences"
}

Rules:
- List at most 4 possibilities, ordered most-to-least consistent with the description.
- Use "could be consistent with" framing implicitly via the reasoning text — never say
  "you have X" or "this is X".
- "higher_but_non_diagnostic" means the described pattern is fairly classic for this
  possibility, but it is still NOT a diagnosis and still requires clinical confirmation.
- red_flags_to_watch_for should list 1-3 warning signs that would mean the person should
  seek urgent/emergency care if they appear (this is a safety net, not the main answer).
- If the description is too vague to say anything useful, return an empty possibilities
  list and use general_advice to ask what additional detail would help.
"""


@dataclass
class SymptomPossibility:
    condition: str
    confidence: str
    reasoning: str


@dataclass
class SymptomCheckResult:
    possibilities: list[SymptomPossibility] = field(default_factory=list)
    red_flags_to_watch_for: list[str] = field(default_factory=list)
    general_advice: str = ""
    available: bool = True

    def as_evidence_text(self) -> str:
        if not self.possibilities:
            return (
                "SYMPTOM CHECK: description was too vague for a useful differential. "
                f"{self.general_advice}"
            )
        lines = ["SYMPTOM CHECK (non-diagnostic, for informational purposes only):"]
        for p in self.possibilities:
            lines.append(f"- Could be consistent with {p.condition} (confidence: {p.confidence}): {p.reasoning}")
        if self.red_flags_to_watch_for:
            lines.append(
                "Seek urgent care if any of these appear: " + "; ".join(self.red_flags_to_watch_for)
            )
        if self.general_advice:
            lines.append(self.general_advice)
        return "\n".join(lines)


class SymptomCheckerAgent:
    def check(self, symptom_description: str, patient_context: str | None = None) -> SymptomCheckResult:
        if llm_client.mock_mode:
            return self._mock_result(symptom_description)

        user_prompt = f"Symptom description: {symptom_description}"
        if patient_context:
            user_prompt += f"\n\nPatient context (self-reported, not clinically verified): {patient_context}"

        try:
            raw = llm_client.complete(SYMPTOM_SYSTEM_PROMPT, user_prompt, json_mode=True, temperature=0.2)
            data = json.loads(raw)
            possibilities = [
                SymptomPossibility(
                    condition=p.get("condition", "unspecified"),
                    confidence=p.get("confidence") if p.get("confidence") in VALID_CONFIDENCE else "low",
                    reasoning=p.get("reasoning", ""),
                )
                for p in data.get("possibilities", [])[:4]
            ]
            return SymptomCheckResult(
                possibilities=possibilities,
                red_flags_to_watch_for=data.get("red_flags_to_watch_for", []) or [],
                general_advice=data.get("general_advice", ""),
            )
        except Exception as e:
            logger.warning(f"Symptom Checker Agent LLM call failed, returning safe fallback: {e}")
            return SymptomCheckResult(
                possibilities=[],
                general_advice=(
                    "I couldn't generate a reliable differential for this description. "
                    "Please describe your symptoms to a doctor or pharmacist directly, "
                    "or seek care if symptoms are severe or worsening."
                ),
                available=False,
            )

    def _mock_result(self, symptom_description: str) -> SymptomCheckResult:
        return SymptomCheckResult(
            possibilities=[
                SymptomPossibility(
                    condition="a common, non-specific cause",
                    confidence="low",
                    reasoning="[MOCK MODE — configure an LLM API key for a real differential]",
                )
            ],
            red_flags_to_watch_for=["symptoms rapidly worsening", "difficulty breathing", "severe pain"],
            general_advice="This is a mock-mode placeholder response; no real symptom analysis was performed.",
        )


symptom_checker_agent = SymptomCheckerAgent()
