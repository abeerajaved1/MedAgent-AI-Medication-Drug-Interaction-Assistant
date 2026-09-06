"""
Core MedAgent orchestration pipeline.

Refactored out of the FastAPI route so both the API (app/main.py) and the
offline evaluation harness (eval/run_eval.py) can call the exact same
logic without going over HTTP.

This revision wires in all of the new agent capabilities on top of the
existing Phase 1-6 pipeline:

  0. Emergency/Triage Agent  — runs FIRST, short-circuits on red flags.
  0.5 Clarification Agent    — runs after routing, short-circuits if a
      safety-relevant fact is missing rather than guessing.
  1. Multi-turn conversational memory — prior turns injected as context
     for the Coordinator and saved again at the end of this turn.
  2. Coordinator (+ new "symptom_check" intent).
  3. Brand-to-generic name resolution — applied to routed medicine names
     before SQL/interaction/RAG lookups.
  4. Tool execution (RAG / SQL / interaction graph / external sources /
     the new Symptom Checker Agent).
  5. Patient profile cross-checks (existing).
  6. Pharmacist Agent draft — optionally via the Self-Consistency Checker
     (draft twice, flag disagreement) instead of a single draft.
  7. Safety/Verifier Agent, now self-consistency-aware, with the existing
     veto -> re-draft loop.
  8. Escalation Agent — explicit, structured "needs a human" signal.
  9. Sentence-level citation attribution.
  10. Full reasoning-trace recording, persisted for GET /api/trace/{id}.
"""
import logging
from dataclasses import dataclass

from app.config import settings
from app.models import ChatResponse, SourceRef, SentenceCitationModel, SymptomPossibilityModel
from app.database import get_patient_profile, get_latest_finding, save_trace, add_history_item
from app.agents.coordinator import coordinator_agent
from app.agents.rag_agent import rag_agent
from app.agents.sql_agent import sql_agent
from app.agents.interaction_tool import drug_interaction_tool
from app.agents.external_source_agent import external_source_agent
from app.agents.patient_profile_agent import patient_profile_agent
from app.agents.dosing_safety_agent import dosing_safety_agent
from app.agents.pharmacist_agent import pharmacist_agent
from app.agents.safety_agent import safety_verifier_agent
from app.agents.emergency_agent import emergency_agent
from app.agents.clarification_agent import clarification_agent
from app.agents.escalation_agent import escalation_agent
from app.agents.consistency_check import self_consistency_checker
from app.agents.symptom_checker_agent import symptom_checker_agent
from app.agents.name_resolution import name_resolution_agent
from app.agents import conversation_memory
from app.citation.attribution import attribute_citations, build_evidence_units
from app.trace.reasoning_trace import new_trace

logger = logging.getLogger("medagent.pipeline")

# Shared severity ranking used to decide whether a newly found concern
# (e.g. a Tier 1 dosing-safety rule) should raise the pipeline's overall
# `severity` variable, which the Safety Agent and Escalation Agent both
# read. "not_found"/None rank alongside "none" here since they don't
# represent an actual risk level to preserve.
SEVERITY_RANK = {None: 0, "not_found": 0, "none": 0, "minor": 1, "moderate": 2, "major": 3}


@dataclass
class PipelineDebugInfo:
    """Extra internals surfaced only for the eval harness — never sent
    over the public API by default."""
    evidence_text: str = ""
    decision_intent: str = ""
    revision_count: int = 0


def run_medagent_pipeline(query: str, session_id: str | None = None,
                           return_debug: bool = False):
    query = query.strip()
    trace = new_trace(query, session_id)

    # ------------------------------------------------------------------
    # Stage 0: Emergency/Triage Agent — always runs first, cannot be
    # skipped by routing logic. Short-circuits the entire pipeline.
    # ------------------------------------------------------------------
    if settings.EMERGENCY_AGENT_ENABLED:
        emergency_result = emergency_agent.check(query)
        trace.log("EmergencyAgent", "check",
                   f"is_emergency={emergency_result.is_emergency}, category={emergency_result.category}")
        if emergency_result.is_emergency:
            response = ChatResponse(
                query_type="emergency",
                explanation=emergency_result.message,
                safety_note="This is an automated safety short-circuit — no other agents were consulted.",
                verification_status="emergency",
                confidence_score=1.0,
                is_emergency=True,
                emergency_category=emergency_result.category,
                escalate_to_human=True,
                escalation_reason="Emergency Agent detected red-flag language.",
                query_id=trace.query_id,
            )
            _finish_trace_and_memory(trace, session_id, query, response)
            return response, (PipelineDebugInfo(decision_intent="emergency") if return_debug else None)

    # ------------------------------------------------------------------
    # Stage 1: conversation memory context (for Coordinator + Pharmacist)
    # ------------------------------------------------------------------
    conversation_context = ""
    has_conversation_history = False
    if settings.CONVERSATION_MEMORY_ENABLED and session_id:
        conversation_context = conversation_memory.build_context_block(
            session_id, limit=settings.CONVERSATION_MEMORY_TURNS
        )
        has_conversation_history = bool(conversation_context)
        trace.log("ConversationMemory", "load_context", f"has_history={has_conversation_history}")

    # ------------------------------------------------------------------
    # Stage 2: Coordinator routing
    # ------------------------------------------------------------------
    decision = coordinator_agent.route(query, conversation_context=conversation_context or None)
    trace.log("CoordinatorAgent", "route",
               f"intent={decision.intent}, medicines={decision.medicines}, tools={decision.required_tools}")

    # ------------------------------------------------------------------
    # Stage 2.5: Adaptive clarification — short-circuits if a
    # safety-relevant fact is missing rather than guessing.
    # ------------------------------------------------------------------
    profile = get_patient_profile(session_id) if session_id else None
    if settings.CLARIFICATION_AGENT_ENABLED:
        clarification = clarification_agent.check(query, decision, profile, has_conversation_history)
        trace.log("ClarificationAgent", "check", f"needs_clarification={clarification.needs_clarification}")
        if clarification.needs_clarification:
            response = ChatResponse(
                query_type=decision.intent,
                medicines=decision.medicines,
                explanation=clarification.question,
                safety_note="Please provide the requested detail so I can answer safely and accurately.",
                verification_status="clarification_needed",
                confidence_score=None,
                needs_clarification=True,
                clarification_reason=clarification.reason,
                query_id=trace.query_id,
            )
            _finish_trace_and_memory(trace, session_id, query, response)
            return response, (PipelineDebugInfo(decision_intent=decision.intent) if return_debug else None)

    # ------------------------------------------------------------------
    # Stage 3: brand-to-generic name resolution (applied before lookups)
    # ------------------------------------------------------------------
    resolved_aliases: dict[str, str] = {}
    lookup_names: dict[str, str] = {m: m for m in decision.medicines}  # display name -> name used for lookups
    if settings.NAME_RESOLUTION_ENABLED and decision.medicines:
        for med in decision.medicines:
            # Only bother resolving if the name isn't already a known local medicine —
            # avoids an unnecessary network round-trip for names we already recognize.
            if med.lower() not in coordinator_agent.known_medicines:
                generic = name_resolution_agent.resolve_generic(med)
                if generic:
                    resolved_aliases[med] = generic
                    lookup_names[med] = generic
        if resolved_aliases:
            trace.log("NameResolutionAgent", "resolve_generic", str(resolved_aliases))

    sources: list[SourceRef] = []
    evidence_parts: list[str] = []
    interaction_status = None
    severity = None
    has_warnings = False
    evidence_found = True

    if resolved_aliases:
        for brand, generic in resolved_aliases.items():
            evidence_parts.append(
                f"Name resolution: '{brand}' is recognized by RxNorm as a brand/alternate name "
                f"for the generic medicine '{generic}'."
            )

    # ------------------------------------------------------------------
    # Stage 4: tool execution based on routing decision
    # ------------------------------------------------------------------
    # Note: avg_rag_score is retained for trace/debugging visibility only.
    # As of the Tier 4 three-factor confidence rewrite, overall
    # confidence_score is driven by citation_coverage (a real per-sentence
    # ratio) rather than raw retrieval similarity — see _compute_confidence().
    avg_rag_score: float | None = None
    if "rag" in decision.required_tools:
        rag_query_medicines = [lookup_names[m] for m in decision.medicines] if decision.medicines else None
        rag_result = rag_agent.retrieve(query, rag_query_medicines)
        if rag_result.documents:
            evidence_parts.append("RAG knowledge base results:\n" + rag_result.as_context_text())
            for d in rag_result.documents:
                sources.append(SourceRef(source=d.title, snippet=d.text[:180], ref_id=d.id))
            avg_rag_score = sum(d.score for d in rag_result.documents) / len(rag_result.documents)
        else:
            evidence_found = False
        trace.log("RAGAgent", "retrieve", f"{len(rag_result.documents)} document(s), avg_score={avg_rag_score}")

    if "sql" in decision.required_tools:
        for med in decision.medicines:
            lookup_name = lookup_names.get(med, med)
            sql_result = sql_agent.lookup_medicine(lookup_name)
            if not sql_result.found and lookup_name != med:
                sql_result = sql_agent.lookup_medicine(med)  # fall back to the original name too
            if sql_result.found:
                evidence_parts.append(f"SQL database record for {med}: {sql_result.medicine_info}")
                sources.append(SourceRef(source=f"Database record: {med}",
                                          snippet=str(sql_result.medicine_info)))
                eml_flag = sql_result.medicine_info.get("on_who_eml")
                if eml_flag is not None:
                    eml_text = (
                        f"{med} is listed on the WHO Model List of Essential Medicines (2023)."
                        if eml_flag else
                        f"{med} is not on the core WHO Model List of Essential Medicines (2023) "
                        f"— regional availability may vary; check local formularies."
                    )
                    evidence_parts.append(eml_text)
                if sql_result.warnings:
                    has_warnings = True
                    for w in sql_result.warnings:
                        evidence_parts.append(f"Safety warning for {med}: {w['description']}")
                        sources.append(SourceRef(source=f"Safety warning: {w['warning_type']}",
                                                  snippet=w["description"][:180]))
        trace.log("SQLAgent", "lookup_medicine", f"{len(decision.medicines)} medicine(s) checked")

    if "interaction_tool" in decision.required_tools and len(decision.medicines) >= 2:
        med_a = lookup_names.get(decision.medicines[0], decision.medicines[0])
        med_b = lookup_names.get(decision.medicines[1], decision.medicines[1])
        check = drug_interaction_tool.check(med_a, med_b)
        if check.status == "found":
            interaction_status = check.severity
            severity = check.severity
            evidence_parts.append(
                f"Drug interaction found between {check.normalized_a} and "
                f"{check.normalized_b} (severity: {check.severity}): {check.description}"
            )
            sources.append(SourceRef(
                source=f"Interaction Graph: {check.normalized_a} + {check.normalized_b}",
                snippet=check.description[:180],
            ))
        elif check.status == "no_known_interaction":
            interaction_status = "none"
            evidence_parts.append(
                f"No known interaction found in the available data between "
                f"{check.normalized_a} and {check.normalized_b}."
            )
        else:
            interaction_status = "not_found"
            evidence_found = False
            evidence_parts.append(
                f"One or both medicines ('{check.drug_a}', '{check.drug_b}') could not be "
                f"matched against the known medicine list."
            )
        trace.log("DrugInteractionTool", "check", f"status={check.status}, severity={severity}")

    if "external" in decision.required_tools and decision.medicines:
        for med in decision.medicines:
            lookup_name = lookup_names.get(med, med)
            ext_result = external_source_agent.lookup_medicine(lookup_name)
            if ext_result.found:
                evidence_found = True
                for d in ext_result.docs:
                    evidence_parts.append(f"{d.source_name}: {d.text}")
                    sources.append(SourceRef(source=d.source_name, snippet=d.text[:180]))

        if "interaction_tool" in decision.required_tools and len(decision.medicines) >= 2:
            ext_hint = external_source_agent.check_interaction_hint(
                lookup_names.get(decision.medicines[0], decision.medicines[0]),
                lookup_names.get(decision.medicines[1], decision.medicines[1]),
            )
            if ext_hint:
                evidence_parts.append(f"{ext_hint.source_name}: {ext_hint.text}")
                sources.append(SourceRef(source=ext_hint.source_name, snippet=ext_hint.text[:180]))
        trace.log("ExternalSourceAgent", "lookup", f"{len(decision.medicines)} medicine(s) checked")

    # New: Symptom Checker Agent (routed via Coordinator's "symptom_check" intent)
    symptom_results: list = []  # collects every SymptomCheckResult that fires, from any source
    if "symptom_checker" in decision.required_tools:
        patient_context_text = None
        if profile:
            from app.agents.reasoning_profile import build_patient_context_block
            patient_context_text = build_patient_context_block(profile) or None
        symptom_result = symptom_checker_agent.check(query, patient_context=patient_context_text)
        symptom_results.append(symptom_result)
        evidence_parts.append(symptom_result.as_evidence_text())
        sources.append(SourceRef(source="Symptom Checker Agent (non-diagnostic)",
                                  snippet=symptom_result.as_evidence_text()[:180]))
        evidence_found = True
        trace.log("SymptomCheckerAgent", "check", f"{len(symptom_result.possibilities)} possibilit(y/ies)")

    # ------------------------------------------------------------------
    # Stage 5: patient profile cross-checks (allergy + polypharmacy)
    # ------------------------------------------------------------------
    profile_flags: list[str] = []
    forced_warning_from_profile = False
    if profile and decision.medicines:
        profile_result = patient_profile_agent.check(profile, decision.medicines)
        if profile_result.has_concerns:
            forced_warning_from_profile = True
            profile_flags = profile_result.allergy_flags + [
                f"Interacts with {h.against_drug} (severity: {h.severity}): {h.description}"
                for h in profile_result.polypharmacy_hits
            ]
            evidence_text_block = patient_profile_agent.as_evidence_text(profile_result)
            evidence_parts.append(f"PATIENT PROFILE CHECK: {evidence_text_block}")
            sources.append(SourceRef(source="Patient profile check", snippet=evidence_text_block[:180]))
            if severity is None:
                severity = "major"
        trace.log("PatientProfileAgent", "check", f"has_concerns={profile_result.has_concerns}")

    # ------------------------------------------------------------------
    # Stage 5.5: Tier 1 rule-based dosing safety checks — age, pregnancy/
    # breastfeeding, renal/hepatic impairment, dosage boundary. Works even
    # without a stored profile (checks inline mentions in the query text).
    # ------------------------------------------------------------------
    dosing_safety_flags: list[str] = []
    if settings.DOSING_SAFETY_ENABLED and decision.medicines:
        dosing_result = dosing_safety_agent.check(profile, decision.medicines, query)
        if dosing_result.has_concerns:
            forced_warning_from_profile = True  # reuses the same escalation wiring as the profile check
            dosing_safety_flags = dosing_result.all_flags()
            evidence_text_block = dosing_safety_agent.as_evidence_text(dosing_result)
            evidence_parts.append(f"DOSING SAFETY RULES: {evidence_text_block}")
            sources.append(SourceRef(source="Dosing safety rules (age/pregnancy/organ/dosage)",
                                      snippet=evidence_text_block[:180]))
            if SEVERITY_RANK.get(dosing_result.severity, 0) > SEVERITY_RANK.get(severity, 0):
                severity = dosing_result.severity
        trace.log("DosingSafetyAgent", "check",
                   f"has_concerns={dosing_result.has_concerns}, severity={dosing_result.severity}")

    # Phase 4: fold in the most recent Vision/OCR Agent finding for this session
    if session_id:
        finding = get_latest_finding(session_id)
        if finding:
            finding_source = finding.get("source", "vision_agent")
            evidence_parts.append(
                f"PRIOR FINDING (preliminary, from {finding_source}, NOT a confirmed "
                f"diagnosis — treat with caution and do not present as fact): {finding['finding_text']}"
            )
            sources.append(SourceRef(
                source=f"{finding_source} (prior finding, this session)",
                snippet=finding["finding_text"][:180],
            ))

            # Tier 2: Vision -> Symptom Checker -> Pharmacist chain. A stored
            # imaging finding is, functionally, a reported symptom/observation
            # — running it through the same Symptom Checker Agent used for
            # user-typed symptoms produces a real differential (with
            # confidence labels) instead of the imaging finding and the
            # medication question just sitting side-by-side as two
            # unconnected evidence blocks. OCR findings (medicine labels,
            # not clinical observations) don't go through this — there's no
            # symptom to check.
            if settings.VISION_SYMPTOM_CHAIN_ENABLED and finding_source == "vision_agent":
                patient_context_text = None
                if profile:
                    from app.agents.reasoning_profile import build_patient_context_block
                    patient_context_text = build_patient_context_block(profile) or None
                vision_symptom_result = symptom_checker_agent.check(
                    finding["finding_text"], patient_context=patient_context_text,
                )
                symptom_results.append(vision_symptom_result)
                evidence_parts.append(
                    "SYMPTOM CHECK DERIVED FROM IMAGING FINDING (chained from the Vision Agent's "
                    "observation above, not a new user-reported symptom):\n"
                    + vision_symptom_result.as_evidence_text()
                )
                sources.append(SourceRef(
                    source="Symptom Checker Agent (chained from imaging finding)",
                    snippet=vision_symptom_result.as_evidence_text()[:180],
                ))
                trace.log("SymptomCheckerAgent", "check_from_vision_finding",
                           f"{len(vision_symptom_result.possibilities)} possibilit(y/ies)")

    if decision.intent == "general" or not evidence_parts:
        evidence_parts.append(
            "No specific medication evidence was retrieved for this query. "
            "It may be a general question, greeting, or a medicine not in the "
            "current knowledge base."
        )
        evidence_found = decision.intent == "general"

    if conversation_context:
        evidence_parts.append(conversation_context)

    evidence_text = "\n\n".join(evidence_parts)

    # ------------------------------------------------------------------
    # Stage 6: Pharmacist Agent draft (optionally via self-consistency)
    # ------------------------------------------------------------------
    self_consistency_flagged = False
    self_consistency_similarity = None
    if settings.SELF_CONSISTENCY_ENABLED and decision.intent != "general":
        consistency_result = self_consistency_checker.check(
            pharmacist_agent, query, evidence_text, patient_profile=profile
        )
        draft = consistency_result.draft_a
        self_consistency_flagged = consistency_result.flagged
        self_consistency_similarity = consistency_result.similarity
        trace.log("SelfConsistencyChecker", "check",
                   f"flagged={self_consistency_flagged}, similarity={self_consistency_similarity}")
    else:
        draft = pharmacist_agent.draft_response(query, evidence_text, patient_profile=profile)
        trace.log("PharmacistAgent", "draft_response", "single draft (self-consistency disabled or general intent)")

    # ------------------------------------------------------------------
    # Stage 7: Safety/Verifier Agent, with veto -> re-draft loop
    # ------------------------------------------------------------------
    revision_count = 0
    verification = safety_verifier_agent.verify(
        user_query=query, evidence_text=evidence_text, draft_response=draft,
        severity=severity, has_warnings=has_warnings or forced_warning_from_profile,
        evidence_found=evidence_found, self_consistency_flagged=self_consistency_flagged,
    )
    trace.log("SafetyAgent", "verify", f"status={verification.status}")

    while verification.status != "approved" and revision_count < settings.MAX_VETO_ROUNDS:
        objection = (
            f"Verification status: {verification.status}. {verification.safety_note} "
            f"Unsupported claims found: {verification.unsupported_claims_found}."
        )
        logger.info(f"Safety Agent vetoed draft (round {revision_count + 1}): {objection}")
        draft = pharmacist_agent.draft_response(query, evidence_text, safety_feedback=objection,
                                                 patient_profile=profile)
        revision_count += 1
        verification = safety_verifier_agent.verify(
            user_query=query, evidence_text=evidence_text, draft_response=draft,
            severity=severity, has_warnings=has_warnings or forced_warning_from_profile,
            evidence_found=evidence_found, self_consistency_flagged=self_consistency_flagged,
        )
        trace.log("SafetyAgent", "re_verify", f"round={revision_count}, status={verification.status}")

    # ------------------------------------------------------------------
    # Stage 8: assemble final structured response
    # ------------------------------------------------------------------
    if verification.status == "insufficient_evidence":
        explanation = (
            "I don't have enough reliable information in my available data to answer "
            "this confidently. " + draft
        )
    else:
        explanation = draft

    # Tier 5 #10: "block vs warn" behavior for major interactions/dosing
    # concerns. Default is the existing behavior (show the answer with a
    # forced warning + escalation flag). Setting BLOCK_ON_MAJOR_SEVERITY=true
    # switches to withholding the drafted explanation entirely in favor of a
    # short, safety-only message when `severity` is "major" — a real
    # implementation of "major interaction -> Reject", not just a wording
    # change, for deployments that want that stricter posture.
    blocked = False
    if settings.BLOCK_ON_MAJOR_SEVERITY and severity == "major":
        blocked = True
        block_reasons = profile_flags + dosing_safety_flags
        if not block_reasons and verification.safety_note:
            # The "major" severity came from the drug interaction check itself
            # rather than a profile/dosing flag — fall back to the Safety
            # Agent's own note so the blocked message still says *why*.
            block_reasons = [verification.safety_note]
        reason_text = (" Specifically: " + "; ".join(block_reasons)) if block_reasons else ""
        explanation = (
            "This question involves a major safety concern, so MedAgent is not providing "
            "a detailed answer automatically." + reason_text +
            " Please consult a licensed doctor or pharmacist before proceeding."
        )
        trace.log("Pipeline", "block_on_major_severity", "explanation replaced with a safety-only message")

    # Stage 8.5: sentence-level citation attribution — moved ahead of
    # confidence scoring (Tier 4) so citation_coverage is available as an
    # input to the three-factor formula below, instead of being computed
    # after confidence already needed it.
    explanation_with_citations = None
    citation_coverage = None
    sentence_citation_models: list[SentenceCitationModel] = []
    if settings.CITATION_ATTRIBUTION_ENABLED and sources and not blocked:
        for i, s in enumerate(sources, start=1):
            s.citation_index = i
        evidence_units = build_evidence_units(sources)
        attribution = attribute_citations(explanation, evidence_units)
        explanation_with_citations = attribution.annotated_text
        citation_coverage = attribution.coverage
        sentence_citation_models = [
            SentenceCitationModel(
                sentence=c.sentence, cited_source_indices=c.cited_indices,
                confidence=c.best_score, supported=c.supported,
            )
            for c in attribution.sentence_citations
        ]
        trace.log("CitationAttribution", "attribute", f"coverage={citation_coverage}")

    confidence_score = _compute_confidence(
        verification_status=verification.status, citation_coverage=citation_coverage,
        self_consistency_similarity=self_consistency_similarity, evidence_found=evidence_found,
    )

    # Stage 8.7: explicit escalate-to-human trigger
    escalation = escalation_agent.evaluate(
        verification_status=verification.status, confidence_score=confidence_score,
        severity=severity, revision_count=revision_count, max_veto_rounds=settings.MAX_VETO_ROUNDS,
        self_consistency_flagged=self_consistency_flagged, profile_has_concerns=forced_warning_from_profile,
    )
    trace.log("EscalationAgent", "evaluate", f"escalate={escalation.escalate}")

    # Merge possibilities/red-flags across every SymptomCheckResult that fired
    # this turn (a user-typed symptom check and/or the vision-chain check
    # above can both contribute) — capped and de-duplicated by condition name
    # so a near-identical hit from both sources doesn't show up twice.
    seen_conditions = set()
    symptom_possibility_models = []
    for sr in symptom_results:
        for p in sr.possibilities:
            key = p.condition.strip().lower()
            if key in seen_conditions:
                continue
            seen_conditions.add(key)
            symptom_possibility_models.append(
                SymptomPossibilityModel(condition=p.condition, confidence=p.confidence, reasoning=p.reasoning)
            )
    symptom_possibility_models = symptom_possibility_models[:6]

    seen_red_flags = set()
    symptom_red_flags = []
    for sr in symptom_results:
        for rf in sr.red_flags_to_watch_for:
            if rf.strip().lower() not in seen_red_flags:
                seen_red_flags.add(rf.strip().lower())
                symptom_red_flags.append(rf)

    response = ChatResponse(
        query_type=decision.intent,
        medicines=decision.medicines,
        interaction_status=interaction_status,
        explanation=explanation,
        safety_note=verification.safety_note,
        verification_status=verification.status,
        confidence_score=confidence_score,
        revision_count=revision_count,
        profile_flags=profile_flags,
        dosing_safety_flags=dosing_safety_flags,
        sources=sources,
        escalate_to_human=escalation.escalate,
        escalation_reason=escalation.reason,
        response_blocked=blocked,
        self_consistency_flagged=self_consistency_flagged,
        self_consistency_similarity=self_consistency_similarity,
        symptom_possibilities=symptom_possibility_models,
        symptom_red_flags=symptom_red_flags,
        explanation_with_citations=explanation_with_citations,
        citation_coverage=citation_coverage,
        sentence_citations=sentence_citation_models,
        query_id=trace.query_id,
        resolved_medicine_aliases=resolved_aliases,
    )

    _finish_trace_and_memory(trace, session_id, query, response)
    if session_id:
        _log_history(session_id, decision.intent, query, response)

    debug = None
    if return_debug:
        debug = PipelineDebugInfo(
            evidence_text=evidence_text, decision_intent=decision.intent, revision_count=revision_count
        )

    return response, debug


HISTORY_TYPE_BY_INTENT = {
    "medicine_info": "medication", "complex": "medication",
    "drug_interaction": "interaction", "symptom_check": "symptoms",
}


def _log_history(session_id: str, intent: str, query: str, response: ChatResponse) -> None:
    """Records this chat turn in the session's History & Saved Results
    list (see app/database.py's history_items table), skipping "general"
    small-talk turns which aren't useful to show in a medication history."""
    item_type = HISTORY_TYPE_BY_INTENT.get(intent)
    if not item_type:
        return
    try:
        title = query if len(query) <= 80 else query[:77] + "..."
        add_history_item(
            session_id=session_id, item_type=item_type, title=title,
            preview=response.explanation, confidence=response.confidence_score,
            verification_status=response.verification_status,
            payload={"query_id": response.query_id, "medicines": response.medicines,
                     "interaction_status": response.interaction_status},
        )
    except Exception as e:
        logger.warning(f"Failed to log history item for session {session_id}: {e}")


def _finish_trace_and_memory(trace, session_id: str | None, query: str, response: ChatResponse) -> None:
    """Persists the reasoning trace (if enabled) and saves this turn to
    conversation memory (if enabled) — shared by every exit path of the
    pipeline (emergency short-circuit, clarification short-circuit, and
    the normal full run)."""
    if settings.REASONING_TRACE_ENABLED:
        try:
            trace.log("Pipeline", "assemble_response", f"verification_status={response.verification_status}")
            save_trace(trace.query_id, session_id, query, trace.to_json())
        except Exception as e:
            logger.warning(f"Failed to persist reasoning trace {trace.query_id}: {e}")

    if settings.CONVERSATION_MEMORY_ENABLED and session_id:
        try:
            from app.agents import conversation_memory as _cm
            _cm.save_turn(session_id, "user", query)
            _cm.save_turn(session_id, "assistant", response.explanation)
        except Exception as e:
            logger.warning(f"Failed to save conversation turn for session {session_id}: {e}")


def _compute_confidence(verification_status: str, citation_coverage: float | None,
                         self_consistency_similarity: float | None, evidence_found: bool) -> float:
    """
    Tier 4 upgrade — three-factor confidence score, replacing the previous
    single heuristic function with the literal formula named in the
    system's own design (c = c_evidence * c_safety * c_model), computed
    from real signals already produced earlier in this same pipeline run
    rather than re-deriving a new proxy:

      - c_evidence: the fraction of the answer's substantive sentences
        that have at least one supported citation, from the sentence-level
        citation attribution feature (app/citation/attribution.py) — a
        real, computed ratio. Falls back to a coarse evidence-found/
        not-found proxy only when citation attribution didn't run at all
        (no sources — e.g. a "general" chit-chat query).
      - c_safety: 1.0 if the Safety/Verifier Agent's final verdict was
        "approved", 0.0 otherwise (hard cutoff, as specified).
      - c_model: the Self-Consistency Checker's agreement score between
        two independently drafted responses (app/agents/consistency_check.py)
        — a free proxy for "model certainty" that costs no extra LLM call
        beyond what self-consistency already spends. Defaults to 1.0 (no
        penalty) when self-consistency didn't run for this query.

    IMPORTANT BEHAVIOR NOTE: because c_safety is a hard 0/1 gate, ANY
    verification_status other than "approved" — including "warning", which
    previously still reported a moderate confidence like 0.6-0.7 under the
    old heuristic — now reports 0% confidence. This is a direct, faithful
    implementation of the "hard 0 otherwise" formula, not a bug, but it is
    a real behavior change worth knowing about: every "warning" response
    will now also cross the Escalation Agent's confidence threshold (see
    app/agents/escalation_agent.py) and escalate, since 0.0 is always below
    ESCALATION_CONFIDENCE_THRESHOLD. If you'd rather keep some signal for
    "moderately risky but still informative" responses instead of flooring
    them to zero, soften this to e.g. `0.5 if verification_status ==
    "warning" else 0.0` — that's a one-line change below.
    """
    c_evidence = citation_coverage if citation_coverage is not None else (0.5 if evidence_found else 0.15)
    c_safety = 1.0 if verification_status == "approved" else 0.0
    c_model = self_consistency_similarity if self_consistency_similarity is not None else 1.0

    return round(c_evidence * c_safety * c_model, 3)
