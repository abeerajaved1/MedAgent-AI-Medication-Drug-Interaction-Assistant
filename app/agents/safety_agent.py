"""
3.6 Safety and Verifier Agent

Final validation layer. Checks the draft response against the retrieved
evidence and structured interaction/warning data, and decides:
  - approved
  - warning_required
  - insufficient_evidence

Combines an LLM-based check (for unsupported claims) with deterministic,
rule-based overrides (for known severe interactions and safety warnings),
so a severe interaction can never silently slip through even if the LLM
verification step fails or is skipped in mock mode.
"""
import json
import logging
from dataclasses import dataclass

from app.llm_client import llm_client

logger = logging.getLogger("medagent.safety_agent")

VERIFY_SYSTEM_PROMPT = """You are the Safety/Verifier Agent, the final validation layer of a
medication information system. You will be given the evidence that was
available and a draft response. Check:
1. Is every claim in the draft supported by the evidence?
2. Does the draft contain any unsupported or fabricated claims?
3. Is the evidence sufficient to answer the user's question reliably?

Return STRICT JSON only, in this exact shape:
{"status": "approved" | "warning" | "insufficient_evidence",
 "unsupported_claims_found": true | false,
 "safety_note": "one short sentence a user should keep in mind"}
"""


@dataclass
class VerificationResult:
    status: str                    # approved | warning | insufficient_evidence
    safety_note: str
    unsupported_claims_found: bool = False
    forced_by_rule: bool = False


class SafetyVerifierAgent:
    def verify(self, user_query: str, evidence_text: str, draft_response: str,
               severity: str | None = None, has_warnings: bool = False,
               evidence_found: bool = True) -> VerificationResult:

        # --- Deterministic rule-based checks first (cannot be bypassed by the LLM) ---
        if not evidence_found:
            return VerificationResult(
                status="insufficient_evidence",
                safety_note="No reliable information was found in the available data for this query. "
                             "Please consult a licensed pharmacist or doctor.",
                forced_by_rule=True,
            )

        if severity == "major":
            return VerificationResult(
                status="warning",
                safety_note="A significant drug interaction was found. Do not combine these "
                             "medicines without first speaking to a doctor or pharmacist.",
                forced_by_rule=True,
            )

        # --- LLM-based verification for unsupported claims / nuanced cases ---
        llm_result = self._llm_verify(user_query, evidence_text, draft_response)

        # Escalate: moderate severity or existing safety warnings should surface as "warning"
        # even if the LLM says approved, per section 3.6 requirements.
        if llm_result.status == "approved" and (severity == "moderate" or has_warnings):
            llm_result.status = "warning"
            if not llm_result.safety_note:
                llm_result.safety_note = "Please review the relevant warnings before use and consult a professional."

        return llm_result

    def _llm_verify(self, user_query: str, evidence_text: str, draft_response: str) -> VerificationResult:
        user_prompt = (
            f"User question: {user_query}\n\n"
            f"--- Evidence ---\n{evidence_text}\n--- End evidence ---\n\n"
            f"--- Draft response ---\n{draft_response}\n--- End draft ---\n\n"
            "Verify the draft now."
        )
        try:
            raw = llm_client.complete(VERIFY_SYSTEM_PROMPT, user_prompt, json_mode=True)
            data = json.loads(raw)
            status = data.get("status", "warning")
            if status not in {"approved", "warning", "insufficient_evidence"}:
                status = "warning"
            return VerificationResult(
                status=status,
                safety_note=data.get("safety_note", "Consult a doctor or pharmacist for personal advice."),
                unsupported_claims_found=bool(data.get("unsupported_claims_found", False)),
            )
        except Exception as e:
            logger.warning(f"Safety LLM verification failed, defaulting to cautious 'warning': {e}")
            return VerificationResult(
                status="warning",
                safety_note="Automatic verification was incomplete — please confirm with a "
                             "pharmacist or doctor before acting on this information.",
                forced_by_rule=True,
            )


safety_verifier_agent = SafetyVerifierAgent()
