"""
Medicine Detail Agent (new — supports the "Ask About Medicine" detail
page's tabbed UI: Uses / Dosage / Side Effects / Contraindications /
Interactions / Warnings / Alternatives).

Aggregates the same sources the chat pipeline uses (SQL record, RAG
knowledge base, external sources) and asks the LLM to organize them
into the exact structured shape the UI needs — grounded strictly in
that evidence, with fields explicitly marked unavailable rather than
guessed when the evidence doesn't cover them (e.g. pharmacokinetic
"Quick Facts" like onset/half-life aren't in the demo dataset for most
drugs, so those fields come back null instead of a fabricated number).
"""
import json
import logging
from dataclasses import dataclass, field

from app.llm_client import llm_client
from app.database import (
    find_medicine, find_warnings, get_age_dosing_rules, get_pregnancy_rule,
    get_organ_impairment_rules, get_dosage_limits,
)
from app.agents.rag_agent import rag_agent
from app.agents.external_source_agent import external_source_agent
from app.graph.interaction_graph import interaction_graph

logger = logging.getLogger("medagent.medicine_detail_agent")

STRUCTURE_SYSTEM_PROMPT = """You are organizing medicine information for a structured UI (tabs: Uses,
Dosage, Side Effects, Contraindications, Warnings, How it works, Quick Facts). Use ONLY the evidence
given — never add a fact, number, or claim that isn't grounded in it. Where the evidence doesn't cover
a field, use null (for Quick Facts values) or an empty list — do not guess plausible-sounding values.

Return STRICT JSON only, in this exact shape:
{"uses": ["short use case", "..."],
 "how_it_works": "one or two plain-language sentences, or null if not in evidence",
 "dosage_forms": ["e.g. Tablets (200 mg, 400 mg)", "..."],
 "dosage_guidance": "plain-language general dosage guidance from the evidence, or null",
 "side_effects_common": ["short phrase", "..."],
 "side_effects_serious": ["short phrase", "..."],
 "contraindications": ["short phrase", "..."],
 "warnings": ["short phrase", "..."],
 "alternatives": ["medicine name", "..."],
 "quick_facts": {"onset_of_action": null, "duration_of_action": null, "peak_effect": null,
                  "bioavailability": null, "half_life": null}
}
Keep every list item short (under ~12 words). Populate quick_facts ONLY if the evidence states it
explicitly; otherwise leave that key null — a null quick fact will be hidden in the UI, which is
preferred over an invented figure.
"""


@dataclass
class QuickFacts:
    onset_of_action: str | None = None
    duration_of_action: str | None = None
    peak_effect: str | None = None
    bioavailability: str | None = None
    half_life: str | None = None
    pregnancy_category: str | None = None
    breastfeeding_category: str | None = None
    max_daily_dose: str | None = None


@dataclass
class MedicineDetail:
    found: bool
    name: str
    generic_name: str | None = None
    drug_class: str | None = None
    on_who_eml: bool | None = None
    uses: list[str] = field(default_factory=list)
    how_it_works: str | None = None
    dosage_forms: list[str] = field(default_factory=list)
    dosage_guidance: str | None = None
    side_effects_common: list[str] = field(default_factory=list)
    side_effects_serious: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    age_restrictions: list[str] = field(default_factory=list)
    known_interactions: list[dict] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    quick_facts: QuickFacts = field(default_factory=QuickFacts)
    confidence_score: float = 0.5
    sources: list[dict] = field(default_factory=list)


class MedicineDetailAgent:
    def get_detail(self, name: str) -> MedicineDetail:
        sql_row = find_medicine(name)
        if not sql_row:
            # Still attempt via RAG/external only — some medicines may exist in
            # the knowledge base/external sources without a local SQL record.
            display_name = name.strip().title()
        else:
            display_name = sql_row["name"]

        evidence_parts = []
        sources = []

        if sql_row:
            evidence_parts.append(
                f"SQL record: name={sql_row['name']}, generic_name={sql_row['generic_name']}, "
                f"category={sql_row['category']}, on_who_eml={bool(sql_row['on_who_eml'])}"
            )
            sources.append({"source": "MedAgent database", "snippet": f"Category: {sql_row['category']}"})

        warnings_rows = find_warnings(display_name)
        for w in warnings_rows:
            evidence_parts.append(f"Safety warning ({w['warning_type']}): {w['description']}")
            sources.append({"source": f"Safety warning: {w['warning_type']}", "snippet": w["description"][:180]})

        rag_result = rag_agent.retrieve(f"{display_name} uses dosage side effects contraindications", [display_name])
        for d in rag_result.documents:
            evidence_parts.append(f"{d.title}: {d.text}")
            sources.append({"source": d.title, "snippet": d.text[:180]})

        ext_result = external_source_agent.lookup_medicine(display_name)
        for d in ext_result.docs:
            evidence_parts.append(f"{d.source_name}: {d.text}")
            sources.append({"source": d.source_name, "snippet": d.text[:180]})

        known_interactions = self._list_known_interactions(display_name)

        # Require a reasonably confident RAG match (not just any nonzero TF-IDF
        # overlap on generic words like "dosage"/"side effects") before treating
        # a name with no SQL record and no external hit as "found" — otherwise
        # a made-up drug name can still surface an unrelated document.
        has_confident_rag_match = any(d.score >= 0.12 for d in rag_result.documents)
        found = bool(sql_row or has_confident_rag_match or ext_result.docs)
        if not found:
            return MedicineDetail(found=False, name=display_name)

        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No detailed evidence available."
        structured = self._structure_with_llm(display_name, evidence_text)

        avg_score = (sum(d.score for d in rag_result.documents) / len(rag_result.documents)
                     if rag_result.documents else 0.5)

        tier1 = self._get_tier1_safety_facts(display_name)

        quick_facts_data = structured.get("quick_facts") or {}
        quick_facts_data["pregnancy_category"] = tier1["pregnancy_category"]
        quick_facts_data["breastfeeding_category"] = tier1["breastfeeding_category"]
        quick_facts_data["max_daily_dose"] = tier1["max_daily_dose"]

        return MedicineDetail(
            found=True,
            name=display_name,
            generic_name=sql_row["generic_name"] if sql_row else None,
            drug_class=sql_row["category"] if sql_row else None,
            on_who_eml=bool(sql_row["on_who_eml"]) if sql_row else None,
            uses=structured.get("uses", []),
            how_it_works=structured.get("how_it_works"),
            dosage_forms=structured.get("dosage_forms", []),
            dosage_guidance=structured.get("dosage_guidance"),
            side_effects_common=structured.get("side_effects_common", []),
            side_effects_serious=structured.get("side_effects_serious", []),
            contraindications=(structured.get("contraindications", []) or []) + tier1["contraindication_notes"],
            warnings=(structured.get("warnings", []) or [w["description"] for w in warnings_rows]) + tier1["warning_notes"],
            age_restrictions=tier1["age_restrictions"],
            known_interactions=known_interactions,
            alternatives=structured.get("alternatives", []),
            quick_facts=QuickFacts(**quick_facts_data),
            confidence_score=round(min(0.97, max(0.4, avg_score if avg_score else 0.6)), 2),
            sources=sources,
        )

    def _format_age_rule(self, rule) -> str:
        """Formats an age_dosing_rules row into a human-readable range,
        handling fractional ages (rules keyed in years, so "under 6 months"
        is stored as max_age=0.5) without printing a confusing "age 0.5"."""
        def fmt(years):
            if years < 1:
                return f"{round(years * 12)} months"
            return f"{years:g} years"

        min_age, max_age = rule["min_age"], rule["max_age"]
        if min_age is not None and max_age is not None:
            span = f"applies age {fmt(min_age)} to {fmt(max_age)}" if min_age > 0 else f"applies under {fmt(max_age)}"
        elif max_age is not None:
            span = f"applies under {fmt(max_age)}"
        elif min_age is not None:
            span = f"applies from age {fmt(min_age)} and up"
        else:
            span = "applies at all ages"
        return f"{rule['description']} ({span})"

    def _get_tier1_safety_facts(self, name: str) -> dict:
        """Deterministic (non-LLM) facts pulled directly from the Tier 1
        rule tables — age dosing rules, pregnancy/breastfeeding category,
        renal/hepatic impairment rules, and the dosage ceiling. These are
        hard data, not LLM-structured evidence, so they're merged in after
        the LLM structuring step rather than being left for the model to
        (possibly inconsistently) restate from free text."""
        age_restrictions = [
            self._format_age_rule(r) for r in get_age_dosing_rules(name)
        ]

        pregnancy_category = None
        breastfeeding_category = None
        contraindication_notes = []
        pregnancy_rule = get_pregnancy_rule(name)
        if pregnancy_rule:
            pregnancy_category = pregnancy_rule["pregnancy_category"]
            breastfeeding_category = pregnancy_rule["breastfeeding_category"]
            if pregnancy_category in ("avoid", "caution"):
                contraindication_notes.append(f"Pregnancy ({pregnancy_category}): {pregnancy_rule['pregnancy_note']}")
            if breastfeeding_category in ("avoid", "caution"):
                contraindication_notes.append(f"Breastfeeding ({breastfeeding_category}): {pregnancy_rule['breastfeeding_note']}")

        warning_notes = [
            f"{r['organ'].capitalize()} impairment ({r['rule_type'].replace('_',' ')}): {r['description']}"
            for r in get_organ_impairment_rules(name)
        ]

        max_daily_dose = None
        limits = get_dosage_limits(name)
        if limits and limits["max_daily_mg"] is not None:
            max_daily_dose = f"{limits['max_daily_mg']:g}mg/day (adult)"
        elif limits:
            max_daily_dose = "Individualized — no fixed daily ceiling"

        return {
            "age_restrictions": age_restrictions,
            "pregnancy_category": pregnancy_category,
            "breastfeeding_category": breastfeeding_category,
            "contraindication_notes": contraindication_notes,
            "warning_notes": warning_notes,
            "max_daily_dose": max_daily_dose,
        }

    def _list_known_interactions(self, name: str) -> list[dict]:
        """Every other drug in the local interaction graph with a known
        direct interaction against `name`, for the detail page's
        'Interactions' tab (distinct from the dedicated pairwise checker)."""
        interaction_graph._ensure_loaded()
        results = []
        for other in interaction_graph.neighbors(name):
            edge = interaction_graph.get_interaction(name, other)
            if edge:
                results.append({"with": other, "severity": edge.severity, "description": edge.description})
        return results

    def _structure_with_llm(self, name: str, evidence_text: str) -> dict:
        if llm_client.mock_mode:
            return {
                "uses": ["[MOCK MODE — configure an LLM API key for structured details]"],
                "how_it_works": None, "dosage_forms": [], "dosage_guidance": None,
                "side_effects_common": [], "side_effects_serious": [], "contraindications": [],
                "warnings": [], "alternatives": [], "quick_facts": {},
            }
        try:
            raw = llm_client.complete(
                STRUCTURE_SYSTEM_PROMPT,
                f"Medicine: {name}\n\n--- Evidence ---\n{evidence_text}\n--- End evidence ---",
                json_mode=True, temperature=0.1,
            )
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Medicine Detail Agent structuring failed for '{name}': {e}")
            return {}


medicine_detail_agent = MedicineDetailAgent()
