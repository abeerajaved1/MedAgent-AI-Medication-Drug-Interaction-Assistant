from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        None, description="Optional session id to apply a stored patient profile (Phase 5), "
                           "conversation memory, and prior vision/OCR findings."
    )


class SourceRef(BaseModel):
    source: str
    snippet: str
    ref_id: Optional[str] = Field(
        None, description="Stable id for feedback loop targeting (RAG doc id); None for non-RAG sources."
    )
    citation_index: Optional[int] = Field(
        None, description="1-based footnote index used by sentence-level citation attribution, "
                           "matching [N] markers in explanation_with_citations."
    )


class SentenceCitationModel(BaseModel):
    sentence: str
    cited_source_indices: list[int] = []
    confidence: float = 0.0
    supported: bool = False


class SymptomPossibilityModel(BaseModel):
    condition: str
    confidence: str   # low | medium | higher_but_non_diagnostic
    reasoning: str


class ChatResponse(BaseModel):
    query_type: str
    medicines: list[str] = []
    interaction_status: Optional[str] = None   # none | minor | moderate | major | not_found
    explanation: str
    safety_note: str
    verification_status: str                   # approved | warning | insufficient_evidence | emergency | clarification_needed
    confidence_score: Optional[float] = None    # 0.0-1.0, heuristic — see pipeline.py _compute_confidence()
    revision_count: int = 0                     # Phase 3: how many times the Safety Agent vetoed the draft
    profile_flags: list[str] = []                # Phase 5: allergy / polypharmacy flags from the patient profile
    dosing_safety_flags: list[str] = Field(
        default_factory=list,
        description="Tier 1 rule-based flags: age-based dosing rules, pregnancy/breastfeeding "
                    "contraindications, renal/hepatic impairment rules, and dosage boundary alerts "
                    "(see app/agents/dosing_safety_agent.py).",
    )
    sources: list[SourceRef] = []
    disclaimer: str = (
        "MedAgent provides general medication information and is not a "
        "substitute for professional medical advice. Always consult a "
        "licensed doctor or pharmacist before making decisions about your "
        "medications."
    )

    # --- New capability fields (all additive; none change existing behavior) ---

    is_emergency: bool = Field(
        False, description="True if the Emergency/Triage Agent short-circuited the pipeline."
    )
    emergency_category: Optional[str] = None

    needs_clarification: bool = Field(
        False, description="True if the Clarification Agent asked a follow-up question instead "
                            "of answering. When true, `explanation` IS the clarifying question."
    )
    clarification_reason: Optional[str] = None

    escalate_to_human: bool = Field(
        False, description="Explicit, structured escalation signal (see app/agents/escalation_agent.py), "
                            "separate from verification_status/confidence_score."
    )
    escalation_reason: Optional[str] = None
    response_blocked: bool = Field(
        False, description="True if BLOCK_ON_MAJOR_SEVERITY is enabled and this response's explanation "
                            "was withheld and replaced with a short safety-only message (see app/pipeline.py)."
    )

    self_consistency_flagged: bool = Field(
        False, description="True if two independently drafted responses disagreed on safety-relevant "
                            "content (see app/agents/consistency_check.py)."
    )
    self_consistency_similarity: Optional[float] = None

    symptom_possibilities: list[SymptomPossibilityModel] = []
    symptom_red_flags: list[str] = []

    explanation_with_citations: Optional[str] = Field(
        None, description="`explanation` with inline [N] footnote markers mapping sentences to "
                           "`sources` entries by citation_index (see app/citation/attribution.py)."
    )
    citation_coverage: Optional[float] = Field(
        None, description="Fraction of substantive sentences in the answer with >=1 supported citation."
    )
    sentence_citations: list[SentenceCitationModel] = []

    query_id: Optional[str] = Field(
        None, description="Id for retrieving the full reasoning trace via GET /api/trace/{query_id}."
    )

    resolved_medicine_aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Brand-name -> generic-name resolutions applied to this query's medicines "
                    "(see app/agents/name_resolution.py), e.g. {'Panadol': 'Acetaminophen'}.",
    )


class PatientProfileRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    age: Optional[int] = Field(None, ge=0, le=130)
    allergies: Optional[list[str]] = None
    conditions: Optional[list[str]] = None
    current_medications: Optional[list[str]] = None


class PatientProfileResponse(BaseModel):
    session_id: str
    age: Optional[int] = None
    allergies: list[str] = []
    conditions: list[str] = []
    current_medications: list[str] = []


class VisionAnalysisResponse(BaseModel):
    available: bool
    findings: str
    disclaimer: str
    saved_to_session: bool = False
    # Tier 2: structured fields from the Vision Agent (see app/agents/vision_agent.py),
    # in addition to `findings` which remains a composed plain-text summary for
    # backward compatibility with any consumer that just wants one string.
    is_xray: bool = True
    finding_location: Optional[str] = None
    finding_description: Optional[str] = None
    confidence_qualifier: Optional[str] = None
    differential_hint: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    rating: str = Field(..., description="'helpful' or 'not_helpful'")
    query_type: Optional[str] = None
    source_ref_ids: list[str] = Field(
        default_factory=list,
        description="ref_id values from the response's sources this feedback applies to (RAG doc ids only).",
    )
    session_id: Optional[str] = None
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackStats(BaseModel):
    total: int
    helpful: int
    not_helpful: int
    by_query_type: dict[str, dict[str, int]]


# --- New: OCR / prescription-label reading ---

class OCRMedicineEntryModel(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None


class OCRResponse(BaseModel):
    available: bool
    medicines: list[OCRMedicineEntryModel] = []
    raw_text: str = ""
    legible: bool = False
    notes: str = ""
    disclaimer: str
    saved_to_session: bool = False


# --- New: standalone symptom-check endpoint (also usable inline via /api/chat) ---

class SymptomCheckRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class SymptomCheckResponse(BaseModel):
    possibilities: list[SymptomPossibilityModel] = []
    red_flags_to_watch_for: list[str] = []
    general_advice: str = ""
    disclaimer: str = (
        "This is a non-diagnostic, informational symptom check only. It cannot replace "
        "an in-person medical evaluation. If symptoms are severe, worsening, or you are "
        "concerned, seek medical care."
    )


# --- New: full reasoning-trace export ---

class TraceResponse(BaseModel):
    query_id: str
    session_id: Optional[str] = None
    query_text: str
    trace: dict
    created_at: Optional[str] = None


# --- New: authentication (powers the Login/Register UI page) ---

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=6, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=1, max_length=200)


class UserModel(BaseModel):
    id: int
    name: str
    email: str
    created_at: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserModel


# --- New: structured medicine detail (powers the "Ask About Medicine" detail page) ---

class QuickFactsModel(BaseModel):
    onset_of_action: Optional[str] = None
    duration_of_action: Optional[str] = None
    peak_effect: Optional[str] = None
    bioavailability: Optional[str] = None
    half_life: Optional[str] = None
    pregnancy_category: Optional[str] = None
    breastfeeding_category: Optional[str] = None
    max_daily_dose: Optional[str] = None


class KnownInteractionModel(BaseModel):
    with_medicine: str
    severity: str
    description: str


class MedicineDetailResponse(BaseModel):
    found: bool
    name: str
    generic_name: Optional[str] = None
    drug_class: Optional[str] = None
    on_who_eml: Optional[bool] = None
    uses: list[str] = []
    how_it_works: Optional[str] = None
    dosage_forms: list[str] = []
    dosage_guidance: Optional[str] = None
    side_effects_common: list[str] = []
    side_effects_serious: list[str] = []
    contraindications: list[str] = []
    warnings: list[str] = []
    age_restrictions: list[str] = []
    known_interactions: list[KnownInteractionModel] = []
    alternatives: list[str] = []
    quick_facts: QuickFactsModel = QuickFactsModel()
    confidence_score: float = 0.5
    sources: list[SourceRef] = []
    disclaimer: str = (
        "This information is for educational purposes only and not a substitute for "
        "professional medical advice or treatment. Always consult a licensed healthcare "
        "professional for medical concerns."
    )


# --- New: structured interaction analysis (powers the Drug Interaction Checker page) ---

class InteractionAnalyzeRequest(BaseModel):
    drug_a: str = Field(..., min_length=1, max_length=120)
    drug_b: str = Field(..., min_length=1, max_length=120)
    session_id: Optional[str] = None


class InteractionAnalyzeResponse(BaseModel):
    status: str    # found | no_known_interaction | not_found
    drug_a: str
    drug_b: str
    severity: Optional[str] = None
    risk_level: Optional[str] = None
    evidence_stars: Optional[int] = None
    why_concern: Optional[str] = None
    possible_effects: list[str] = []
    what_you_should_do: list[str] = []
    seek_care_if: Optional[str] = None
    sources: list[dict] = []
    disclaimer: str = (
        "MedAgent checks interactions using trusted databases and clinical guidelines. "
        "Results may not include all possible interactions — always consult a licensed "
        "healthcare professional before combining medications."
    )


# --- New: history & saved results (powers Dashboard + History page) ---

class HistoryItemModel(BaseModel):
    id: int
    session_id: str
    item_type: str
    title: str
    preview: str
    confidence: Optional[float] = None
    verification_status: Optional[str] = None
    saved: bool
    created_at: str


class HistoryListResponse(BaseModel):
    items: list[HistoryItemModel]
    counts_by_type: dict[str, int]
    total: int


class DashboardResponse(BaseModel):
    recent_activity: list[HistoryItemModel]
    saved_results: list[HistoryItemModel]
    safety_alerts: list[dict]
