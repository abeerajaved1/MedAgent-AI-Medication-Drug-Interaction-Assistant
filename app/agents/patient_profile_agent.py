"""
Phase 5: Patient Profile Agent

Not one of the original 6 agents — a new, small rule-based agent that
cross-checks a session's stored patient profile (age, allergies,
conditions, current medications) against whatever medicine(s) are being
discussed, and against the interaction graph for polypharmacy hits.

This produces deterministic, rule-based flags (not LLM judgment) that
the Safety/Verifier Agent treats as hard evidence — an allergy match or
a polypharmacy interaction hit always forces at least a "warning" status.
"""
import logging
from dataclasses import dataclass, field

from app.database import find_warnings
from app.agents.interaction_tool import drug_interaction_tool, PolypharmacyHit

logger = logging.getLogger("medagent.patient_profile_agent")


@dataclass
class ProfileCheckResult:
    allergy_flags: list[str] = field(default_factory=list)
    polypharmacy_hits: list[PolypharmacyHit] = field(default_factory=list)

    @property
    def has_concerns(self) -> bool:
        return bool(self.allergy_flags or self.polypharmacy_hits)


class PatientProfileAgent:
    def check(self, profile: dict | None, medicines: list[str]) -> ProfileCheckResult:
        result = ProfileCheckResult()
        if not profile or not medicines:
            return result

        allergies = [a.lower() for a in profile.get("allergies", [])]
        current_meds = profile.get("current_medications", [])

        for med in medicines:
            # Direct name match against stated allergies
            if med.lower() in allergies:
                result.allergy_flags.append(
                    f"Patient profile lists an allergy to {med} directly."
                )
            # Cross-check the medicine's own "Allergic Reaction" type warnings
            # against stated allergies (e.g. allergy to "penicillin" + Amoxicillin).
            for w in find_warnings(med):
                if w["warning_type"].lower() in ("allergic reaction",) or "allerg" in w["warning_type"].lower():
                    for allergy in allergies:
                        if allergy and allergy in w["description"].lower():
                            result.allergy_flags.append(
                                f"{med} carries an allergy warning that matches the patient's "
                                f"stated allergy to '{allergy}': {w['description']}"
                            )

            # Polypharmacy: does this medicine interact with anything already
            # on the patient's current medication list?
            hits = drug_interaction_tool.check_against_medication_list(med, current_meds)
            result.polypharmacy_hits.extend(hits)

        if result.has_concerns:
            logger.info(f"Patient profile check found concerns: "
                        f"{len(result.allergy_flags)} allergy flag(s), "
                        f"{len(result.polypharmacy_hits)} polypharmacy hit(s).")
        return result

    def as_evidence_text(self, result: ProfileCheckResult) -> str | None:
        if not result.has_concerns:
            return None
        parts = []
        for flag in result.allergy_flags:
            parts.append(f"ALLERGY FLAG: {flag}")
        for hit in result.polypharmacy_hits:
            parts.append(
                f"POLYPHARMACY FLAG: interacts with {hit.against_drug} already on the "
                f"patient's medication list (severity: {hit.severity}): {hit.description}"
            )
        return "\n".join(parts)


patient_profile_agent = PatientProfileAgent()
