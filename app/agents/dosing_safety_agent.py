"""
Dosing Safety Agent — Tier 1 rule-based safety checks.

Four deterministic, rule-table-driven checks that were previously named
in the system's design but never actually implemented:

  1. Age-based dosing rules       (age_dosing_rules table)
  2. Pregnancy/breastfeeding      (pregnancy_rules table)
  3. Renal/hepatic impairment     (organ_impairment_rules table)
  4. Dosage boundary checking     (medicine_dosage_limits table + text extraction)

Like the Patient Profile Agent (app/agents/patient_profile_agent.py), this
produces deterministic flags — not LLM judgment — that the Safety/Verifier
Agent treats as hard evidence. An "avoid" rule or an over-limit dosage
mention always forces at least a "warning" verdict.

Inputs are read from two places, so these checks work even for guests who
never filled in a stored patient profile:
  - The stored patient profile (age, conditions) when a session_id is given.
  - The query text itself (inline age mentions, "pregnant"/"breastfeeding"
    language, "kidney disease"/"liver disease" mentions, explicit dosage
    mentions like "8 ibuprofen tablets" or "4000mg").

Honest limitation: dosage-mention extraction is regex-based pattern
matching, not a clinical NLP parser. It handles the common phrasings
("N tablets/pills/capsules", "Nmg") but will miss unusual phrasing and
can occasionally misread a stated product strength (e.g. "ibuprofen
200mg tablets") as a total dose — treat its flags as a prompt to look
closer, not as a certified dose calculation.
"""
import logging
import re
from dataclasses import dataclass, field

from app.database import (
    get_age_dosing_rules, get_pregnancy_rule, get_organ_impairment_rules, get_dosage_limits,
)

logger = logging.getLogger("medagent.dosing_safety_agent")

# --- text-extraction patterns ---------------------------------------------

INLINE_AGE_PATTERNS = [
    (re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?|y/?o)\s*old\b"), 1.0),
    (re.compile(r"\bage(?:d)?\s*(\d{1,3})\b"), 1.0),
    (re.compile(r"\b(\d{1,2})\s*months?\s*old\b"), 1 / 12),
]

PREGNANCY_PATTERN = re.compile(r"\bpregnan(t|cy)\b")
BREASTFEEDING_PATTERN = re.compile(r"\bbreast[\s-]?feed(ing)?\b|\bnursing\b|\bbreast\s*milk\b")
RENAL_PATTERN = re.compile(r"\bkidney (disease|failure|impairment|problem)s?\b|\brenal (impairment|failure|disease)\b")
HEPATIC_PATTERN = re.compile(r"\bliver (disease|failure|impairment|problem)s?\b|\bhepatic (impairment|failure|disease)\b")

UNIT_QUANTITY_PATTERN = re.compile(r"\b(\d{1,3})\s+(?:[a-zA-Z]+\s+){0,1}(tablets?|pills?|capsules?|caps?)\b")
TOTAL_MG_PATTERN = re.compile(r"\b(\d{2,5})\s*mg\b")

SEVERITY_ORDER = {"none": 0, "minor": 1, "moderate": 2, "major": 3}
RULE_TYPE_TO_SEVERITY = {"avoid": "major", "dose_adjust": "moderate", "caution": "moderate", "monitor": "minor"}


def extract_inline_age(text: str) -> float | None:
    for pattern, unit_years in INLINE_AGE_PATTERNS:
        m = pattern.search(text.lower())
        if m:
            return round(int(m.group(1)) * unit_years, 2)
    return None


def mentions_pregnancy(text: str) -> bool:
    return bool(PREGNANCY_PATTERN.search(text.lower()))


def mentions_breastfeeding(text: str) -> bool:
    return bool(BREASTFEEDING_PATTERN.search(text.lower()))


def mentions_renal_impairment(text: str) -> bool:
    return bool(RENAL_PATTERN.search(text.lower()))


def mentions_hepatic_impairment(text: str) -> bool:
    return bool(HEPATIC_PATTERN.search(text.lower()))


@dataclass
class DosageMention:
    quantity_units: int | None = None
    total_mg: int | None = None
    raw: str = ""


def extract_dosage_mention(text: str) -> DosageMention | None:
    """Best-effort extraction of a stated dose from free text. Prefers an
    explicit unit count ("8 tablets") since that's unambiguous; falls back
    to a bare mg number only if no unit-count phrasing is present, since a
    bare mg number is more likely to be describing a product's per-unit
    strength than a total consumed (still an approximation either way —
    see the module docstring)."""
    text_lower = text.lower()
    m = UNIT_QUANTITY_PATTERN.search(text_lower)
    if m:
        return DosageMention(quantity_units=int(m.group(1)), raw=m.group(0))
    m = TOTAL_MG_PATTERN.search(text_lower)
    if m:
        return DosageMention(total_mg=int(m.group(1)), raw=m.group(0))
    return None


@dataclass
class DosingSafetyResult:
    age_flags: list[str] = field(default_factory=list)
    pregnancy_flags: list[str] = field(default_factory=list)
    breastfeeding_flags: list[str] = field(default_factory=list)
    organ_flags: list[str] = field(default_factory=list)
    dosage_flags: list[str] = field(default_factory=list)
    severity: str = "none"   # none | minor | moderate | major — the highest severity among all triggered flags

    @property
    def has_concerns(self) -> bool:
        return bool(self.age_flags or self.pregnancy_flags or self.breastfeeding_flags
                    or self.organ_flags or self.dosage_flags)

    def all_flags(self) -> list[str]:
        return self.age_flags + self.pregnancy_flags + self.breastfeeding_flags + self.organ_flags + self.dosage_flags

    def _bump_severity(self, candidate: str):
        if SEVERITY_ORDER.get(candidate, 0) > SEVERITY_ORDER.get(self.severity, 0):
            self.severity = candidate


class DosingSafetyAgent:
    def check(self, profile: dict | None, medicines: list[str], query_text: str) -> DosingSafetyResult:
        result = DosingSafetyResult()
        if not medicines:
            return result

        # --- resolve age (stored profile takes precedence over inline mention) ---
        age = (profile or {}).get("age")
        if age is None:
            age = extract_inline_age(query_text)

        # --- resolve pregnancy/breastfeeding/organ-impairment signals ---
        conditions_text = " ".join((profile or {}).get("conditions", [])).lower()
        is_pregnant = mentions_pregnancy(query_text) or "pregnan" in conditions_text
        is_breastfeeding = mentions_breastfeeding(query_text) or "breastfeed" in conditions_text or "nursing" in conditions_text
        has_renal_impairment = mentions_renal_impairment(query_text) or mentions_renal_impairment(conditions_text)
        has_hepatic_impairment = mentions_hepatic_impairment(query_text) or mentions_hepatic_impairment(conditions_text)

        dosage_mention = extract_dosage_mention(query_text)

        for med in medicines:
            if age is not None:
                self._check_age(med, age, result)
            if is_pregnant:
                self._check_pregnancy(med, result)
            if is_breastfeeding:
                self._check_breastfeeding(med, result)
            if has_renal_impairment:
                self._check_organ(med, "renal", result)
            if has_hepatic_impairment:
                self._check_organ(med, "hepatic", result)
            if dosage_mention:
                # Guard against misattributing a dosage mention to the wrong
                # medicine when multiple medicines are named in one query
                # (e.g. "8 ibuprofen tablets with my aspirin" — the "8" is
                # about Ibuprofen, not Aspirin). Only apply the check when
                # the medicine is unambiguous: either it's the only medicine
                # in the query, or its name appears in the matched phrase.
                unambiguous = len(medicines) == 1 or med.lower() in dosage_mention.raw.lower()
                if unambiguous:
                    self._check_dosage(med, dosage_mention, result)

        if result.has_concerns:
            logger.info(f"Dosing Safety Agent found concerns (severity={result.severity}): "
                        f"{len(result.all_flags())} flag(s).")
        return result

    def _check_age(self, med: str, age: float, result: DosingSafetyResult):
        for rule in get_age_dosing_rules(med):
            min_age = rule["min_age"]
            max_age = rule["max_age"]
            in_range = (min_age is None or age >= min_age) and (max_age is None or age <= max_age)
            if in_range:
                age_display = f"{round(age * 12)} months" if age < 1 else f"{age:g} years"
                result.age_flags.append(
                    f"AGE RULE ({med}, age {age_display}): {rule['description']}"
                )
                result._bump_severity(RULE_TYPE_TO_SEVERITY.get(rule["rule_type"], "moderate"))

    def _check_pregnancy(self, med: str, result: DosingSafetyResult):
        rule = get_pregnancy_rule(med)
        if rule and rule["pregnancy_category"] in ("avoid", "caution"):
            result.pregnancy_flags.append(
                f"PREGNANCY FLAG ({med}, category: {rule['pregnancy_category']}): {rule['pregnancy_note']}"
            )
            result._bump_severity(RULE_TYPE_TO_SEVERITY.get(
                "avoid" if rule["pregnancy_category"] == "avoid" else "caution", "moderate"
            ))

    def _check_breastfeeding(self, med: str, result: DosingSafetyResult):
        rule = get_pregnancy_rule(med)
        if rule and rule["breastfeeding_category"] in ("avoid", "caution"):
            result.breastfeeding_flags.append(
                f"BREASTFEEDING FLAG ({med}, category: {rule['breastfeeding_category']}): {rule['breastfeeding_note']}"
            )
            result._bump_severity(RULE_TYPE_TO_SEVERITY.get(
                "avoid" if rule["breastfeeding_category"] == "avoid" else "caution", "moderate"
            ))

    def _check_organ(self, med: str, organ: str, result: DosingSafetyResult):
        for rule in get_organ_impairment_rules(med, organ):
            result.organ_flags.append(
                f"{organ.upper()} IMPAIRMENT FLAG ({med}, {rule['rule_type'].replace('_',' ')}): {rule['description']}"
            )
            result._bump_severity(RULE_TYPE_TO_SEVERITY.get(rule["rule_type"], "moderate"))

    def _check_dosage(self, med: str, mention: DosageMention, result: DosingSafetyResult):
        limits = get_dosage_limits(med)
        if not limits or limits["max_daily_mg"] is None:
            return  # no fixed ceiling to check against (e.g. individualized dosing like Warfarin)

        total_mg = None
        if mention.quantity_units is not None and limits["unit_dose_mg"]:
            total_mg = mention.quantity_units * limits["unit_dose_mg"]
        elif mention.total_mg is not None:
            total_mg = mention.total_mg

        if total_mg is None:
            return

        max_daily = limits["max_daily_mg"]
        if total_mg >= max_daily:
            result.dosage_flags.append(
                f"DOSAGE ALERT ({med}): mentioned amount (~{total_mg:g}mg, from \"{mention.raw}\") is at or "
                f"above the typical adult daily maximum of {max_daily:g}mg. {limits['notes'] or ''}".strip()
            )
            result._bump_severity("major")
        elif total_mg >= 0.75 * max_daily:
            result.dosage_flags.append(
                f"DOSAGE CAUTION ({med}): mentioned amount (~{total_mg:g}mg, from \"{mention.raw}\") is "
                f"approaching the typical adult daily maximum of {max_daily:g}mg. {limits['notes'] or ''}".strip()
            )
            result._bump_severity("moderate")

    def as_evidence_text(self, result: DosingSafetyResult) -> str | None:
        if not result.has_concerns:
            return None
        return "\n".join(result.all_flags())


dosing_safety_agent = DosingSafetyAgent()
