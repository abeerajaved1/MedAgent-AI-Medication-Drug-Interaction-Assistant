"""
MedAgent — main FastAPI application.

Orchestration logic lives in app/pipeline.py (shared with the eval
harness). This file just wires up the API routes.

New in this revision:
  - POST /api/ocr           — Prescription/Label OCR Agent
  - POST /api/symptom-check — standalone Symptom Checker Agent (the same
    agent also runs inline via /api/chat when the Coordinator detects a
    "symptom_check" intent)
  - GET  /api/trace/{query_id} — full reasoning-trace export
"""
import logging
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import (
    init_db, upsert_patient_profile, get_patient_profile, save_finding,
    save_feedback, get_feedback_stats, get_trace, list_all_medicine_names,
    list_history, count_history_by_type, toggle_history_saved, delete_history_item,
    find_warnings,
)
from app.models import (
    ChatRequest, ChatResponse, PatientProfileRequest, PatientProfileResponse,
    VisionAnalysisResponse, FeedbackRequest, FeedbackStats,
    OCRResponse, OCRMedicineEntryModel, SymptomCheckRequest, SymptomCheckResponse,
    SymptomPossibilityModel, TraceResponse, RegisterRequest, LoginRequest, AuthResponse,
    UserModel, MedicineDetailResponse, KnownInteractionModel, QuickFactsModel,
    InteractionAnalyzeRequest, InteractionAnalyzeResponse, HistoryItemModel,
    HistoryListResponse, DashboardResponse, SourceRef,
)
from app.pipeline import run_medagent_pipeline
from app.agents.vision_agent import vision_agent
from app.agents.ocr_agent import ocr_agent
from app.agents.symptom_checker_agent import symptom_checker_agent
from app.agents.medicine_detail_agent import medicine_detail_agent
from app.agents.interaction_detail_agent import interaction_detail_agent
from app import auth as auth_module

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("medagent.main")

app = FastAPI(title="MedAgent", version="2.1.0",
              description="AI Agent-based medication information and drug interaction checker.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    from app.graph.interaction_graph import interaction_graph
    interaction_graph._ensure_loaded()

    # Tier 3 #8: optional real TWOSIDES data load, persisted across every
    # app start (not just a one-off script run) when TWOSIDES_CSV_PATH is
    # set. The file itself must be downloaded by you — see
    # scripts/load_twosides.py for why this can't happen automatically in
    # this delivery (the free public dataset is hosted on infrastructure
    # outside this build environment's network access and is too large to
    # ship in a delivery zip).
    if settings.TWOSIDES_CSV_PATH:
        import os as _os
        if _os.path.exists(settings.TWOSIDES_CSV_PATH):
            before = interaction_graph.graph.number_of_nodes(), interaction_graph.graph.number_of_edges()
            csv_path = settings.TWOSIDES_CSV_PATH
            if csv_path.endswith(".gz"):
                import gzip, shutil, tempfile
                tmp_path = tempfile.mkstemp(suffix=".csv")[1]
                with gzip.open(csv_path, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                csv_path = tmp_path
            added = interaction_graph.load_twosides_csv(csv_path, max_rows=settings.TWOSIDES_MAX_ROWS)
            after = interaction_graph.graph.number_of_nodes(), interaction_graph.graph.number_of_edges()
            logger.info(f"TWOSIDES data loaded from {settings.TWOSIDES_CSV_PATH}: {added} rows processed, "
                        f"graph grew from {before[0]} nodes/{before[1]} edges to {after[0]} nodes/{after[1]} edges.")
        else:
            logger.warning(f"TWOSIDES_CSV_PATH is set to '{settings.TWOSIDES_CSV_PATH}' but that file doesn't "
                            f"exist — skipping TWOSIDES import. See scripts/load_twosides.py for how to get one.")

    logger.info(f"MedAgent started. LLM provider = {settings.LLM_PROVIDER}")


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER}


# ---------------------------------------------------------------------
# New: authentication. The session token doubles as `session_id`
# everywhere else in the app (chat memory, profile, history), so the
# frontend just stores this one token and passes it as session_id.
# ---------------------------------------------------------------------

def get_current_user(authorization: str | None = Header(None)) -> dict | None:
    """Optional-auth dependency: returns the user dict if a valid
    'Authorization: Bearer <token>' header is present, else None. Routes
    that require login check for None themselves rather than this
    dependency raising, since most of MedAgent's features (chat, medicine
    lookup) work fine for anonymous/guest session_ids too."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return auth_module.get_user_from_token(token)


@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    try:
        result = auth_module.register(req.name, req.email, req.password)
    except auth_module.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AuthResponse(token=result["token"], user=UserModel(**result["user"]))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    try:
        result = auth_module.login(req.email, req.password)
    except auth_module.AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return AuthResponse(token=result["token"], user=UserModel(**result["user"]))


@app.get("/api/auth/me", response_model=UserModel)
def me(user: dict | None = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return UserModel(**user)


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(None)):
    if authorization and authorization.lower().startswith("bearer "):
        auth_module.logout(authorization.split(" ", 1)[1].strip())
    return {"status": "logged_out"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response, _ = run_medagent_pipeline(req.message, session_id=req.session_id)
    return response


@app.post("/api/profile", response_model=PatientProfileResponse)
def set_profile(req: PatientProfileRequest):
    profile = upsert_patient_profile(
        session_id=req.session_id, age=req.age, allergies=req.allergies,
        conditions=req.conditions, current_medications=req.current_medications,
    )
    return PatientProfileResponse(**profile)


@app.get("/api/profile/{session_id}", response_model=PatientProfileResponse)
def read_profile(session_id: str):
    profile = get_patient_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found for this session_id.")
    return PatientProfileResponse(**profile)


@app.post("/api/vision", response_model=VisionAnalysisResponse)
async def analyze_image(file: UploadFile = File(...), session_id: str | None = Form(None)):
    """
    Phase 4: preliminary chest X-ray analysis (Vision Agent).
    If session_id is given, the finding is stored and will be pulled into
    later /api/chat calls in the same session for combined synthesis.
    """
    image_bytes = await file.read()
    result = vision_agent.analyze_xray(image_bytes, mime_type=file.content_type or "image/jpeg")

    saved = False
    if session_id and result.available:
        save_finding(session_id, result.findings, source="vision_agent")
        from app.database import add_history_item
        add_history_item(
            session_id=session_id, item_type="image_analysis", title="Medical image analysis",
            preview=result.findings, confidence=None, verification_status="approved",
        )
        saved = True

    return VisionAnalysisResponse(
        available=result.available, findings=result.findings,
        is_xray=result.is_xray, finding_location=result.finding_location,
        finding_description=result.finding_description, confidence_qualifier=result.confidence_qualifier,
        differential_hint=result.differential_hint,
        disclaimer=result.disclaimer, saved_to_session=saved,
    )


@app.post("/api/ocr", response_model=OCRResponse)
async def read_prescription_label(file: UploadFile = File(...), session_id: str | None = Form(None)):
    """
    New capability: Prescription/Label OCR Agent. Reads a photographed
    pill bottle label or prescription slip and extracts medicine
    name(s) + dosage + frequency. Reuses the same Gemini multimodal
    endpoint as the Vision Agent (see app/agents/ocr_agent.py).

    If session_id is given, the extraction is stored as a session
    finding and will be pulled into later /api/chat calls in the same
    session, the same way Vision Agent findings are.
    """
    image_bytes = await file.read()
    result = ocr_agent.read_label(image_bytes, mime_type=file.content_type or "image/jpeg")

    saved = False
    if session_id and result.available and result.medicines:
        save_finding(session_id, ocr_agent.as_finding_text(result), source="ocr_agent")
        from app.database import add_history_item
        add_history_item(
            session_id=session_id, item_type="prescription", title="Prescription/label scan",
            preview=ocr_agent.as_finding_text(result), confidence=None, verification_status="approved",
        )
        saved = True

    return OCRResponse(
        available=result.available,
        medicines=[OCRMedicineEntryModel(name=m.name, dosage=m.dosage, frequency=m.frequency)
                   for m in result.medicines],
        raw_text=result.raw_text,
        legible=result.legible,
        notes=result.notes,
        disclaimer=result.disclaimer,
        saved_to_session=saved,
    )


@app.post("/api/symptom-check", response_model=SymptomCheckResponse)
def symptom_check(req: SymptomCheckRequest):
    """
    New capability: standalone Symptom Checker Agent endpoint. The same
    agent also runs automatically inside /api/chat when the Coordinator
    classifies a message as a "symptom_check" intent — this endpoint is
    for callers who want the differential directly without the full
    chat pipeline (e.g. a dedicated "check my symptoms" UI panel).
    """
    patient_context = None
    if req.session_id:
        profile = get_patient_profile(req.session_id)
        if profile:
            from app.agents.reasoning_profile import build_patient_context_block
            patient_context = build_patient_context_block(profile) or None

    result = symptom_checker_agent.check(req.description, patient_context=patient_context)
    return SymptomCheckResponse(
        possibilities=[
            SymptomPossibilityModel(condition=p.condition, confidence=p.confidence, reasoning=p.reasoning)
            for p in result.possibilities
        ],
        red_flags_to_watch_for=result.red_flags_to_watch_for,
        general_advice=result.general_advice,
    )


@app.get("/api/trace/{query_id}", response_model=TraceResponse)
def get_reasoning_trace(query_id: str):
    """
    New capability: full reasoning-trace export. Returns the complete
    agent-by-agent trace for a single /api/chat call — which tools
    fired, what each contributed, and every safety veto/revision round
    — supporting the system's "transparency by design" claim.
    """
    import json
    row = get_trace(query_id)
    if not row:
        raise HTTPException(status_code=404, detail="No trace found for this query_id.")
    return TraceResponse(
        query_id=row["query_id"], session_id=row["session_id"], query_text=row["query_text"],
        trace=json.loads(row["trace_json"]), created_at=row.get("created_at"),
    )


@app.get("/api/medicines")
def list_medicines(q: str | None = None):
    """Powers the medicine-name autocomplete used on the Drug Interaction
    Checker and the global search bar."""
    names = list_all_medicine_names()
    if q:
        q_lower = q.lower()
        names = [n for n in names if q_lower in n.lower()]
    return {"medicines": sorted(names)}


@app.get("/api/medicines/{name}", response_model=MedicineDetailResponse)
def get_medicine_detail(name: str):
    """New capability: structured medicine detail for the "Ask About
    Medicine" detail page's tabbed UI (Uses/Dosage/Side Effects/
    Contraindications/Warnings/Alternatives)."""
    detail = medicine_detail_agent.get_detail(name)
    if not detail.found:
        raise HTTPException(status_code=404, detail=f"No information found for '{name}'.")
    return MedicineDetailResponse(
        found=True, name=detail.name, generic_name=detail.generic_name, drug_class=detail.drug_class,
        on_who_eml=detail.on_who_eml, uses=detail.uses, how_it_works=detail.how_it_works,
        dosage_forms=detail.dosage_forms, dosage_guidance=detail.dosage_guidance,
        side_effects_common=detail.side_effects_common, side_effects_serious=detail.side_effects_serious,
        contraindications=detail.contraindications, warnings=detail.warnings,
        age_restrictions=detail.age_restrictions,
        known_interactions=[
            KnownInteractionModel(with_medicine=i["with"], severity=i["severity"], description=i["description"])
            for i in detail.known_interactions
        ],
        alternatives=detail.alternatives, quick_facts=QuickFactsModel(**detail.quick_facts.__dict__),
        confidence_score=detail.confidence_score,
        sources=[SourceRef(source=s["source"], snippet=s["snippet"]) for s in detail.sources],
    )


@app.post("/api/interactions/analyze", response_model=InteractionAnalyzeResponse)
def analyze_interaction(req: InteractionAnalyzeRequest):
    """New capability: structured interaction analysis for the Drug
    Interaction Checker page (risk level, evidence strength, why-a-
    concern explanation, possible effects, what-you-should-do guidance)."""
    result = interaction_detail_agent.analyze(req.drug_a, req.drug_b)
    response = InteractionAnalyzeResponse(
        status=result.status, drug_a=result.drug_a, drug_b=result.drug_b, severity=result.severity,
        risk_level=result.risk_level, evidence_stars=result.evidence_stars, why_concern=result.why_concern,
        possible_effects=result.possible_effects, what_you_should_do=result.what_you_should_do,
        seek_care_if=result.seek_care_if, sources=result.sources,
    )
    if req.session_id and result.status in ("found", "no_known_interaction"):
        from app.database import add_history_item
        confidence = 0.9 if result.status == "found" else 0.7
        add_history_item(
            session_id=req.session_id, item_type="interaction",
            title=f"{result.drug_a} + {result.drug_b} interaction",
            preview=result.why_concern or "No known interaction found.",
            confidence=confidence,
            verification_status=("warning" if result.severity in ("major", "moderate") else "approved"),
            payload=response.model_dump(),
        )
    return response


@app.get("/api/history/{session_id}", response_model=HistoryListResponse)
def get_history(session_id: str, type: str | None = None, saved_only: bool = False,
                 limit: int = 50, offset: int = 0):
    """New capability: powers the History & Saved Results page and the
    Dashboard's Recent Activity / Saved Results panels."""
    items = list_history(session_id, item_type=type, saved_only=saved_only, limit=limit, offset=offset)
    counts = count_history_by_type(session_id)
    return HistoryListResponse(
        items=[HistoryItemModel(**{**i, "saved": bool(i["saved"])}) for i in items],
        counts_by_type=counts, total=counts.get("all", 0),
    )


@app.post("/api/history/{item_id}/toggle-save")
def toggle_save(item_id: int, session_id: str):
    updated = toggle_history_saved(item_id, session_id)
    if not updated:
        raise HTTPException(status_code=404, detail="History item not found for this session.")
    return {"id": item_id, "saved": bool(updated["saved"])}


@app.delete("/api/history/{item_id}")
def remove_history_item(item_id: int, session_id: str):
    ok = delete_history_item(item_id, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="History item not found for this session.")
    return {"status": "deleted"}


@app.get("/api/dashboard/{session_id}", response_model=DashboardResponse)
def dashboard(session_id: str):
    """New capability: single call powering the Dashboard page's Recent
    Activity, Saved Results, and Safety Alerts panels."""
    recent = list_history(session_id, limit=6)
    saved = list_history(session_id, saved_only=True, limit=6)

    safety_alerts = []
    for item in list_history(session_id, item_type="interaction", limit=20):
        if item.get("verification_status") == "warning":
            safety_alerts.append({
                "level": "high_risk", "title": "High Risk Interaction",
                "description": item["preview"][:160], "created_at": item["created_at"],
            })
    profile = get_patient_profile(session_id)
    if profile and profile.get("current_medications"):
        for med in profile["current_medications"]:
            for w in find_warnings(med):
                safety_alerts.append({
                    "level": "warning", "title": f"{med}: {w['warning_type']}",
                    "description": w["description"][:160], "created_at": None,
                })
    safety_alerts.append({
        "level": "info", "title": "General Reminder",
        "description": "This tool provides AI-assisted information only.", "created_at": None,
    })

    return DashboardResponse(
        recent_activity=[HistoryItemModel(**{**i, "saved": bool(i["saved"])}) for i in recent],
        saved_results=[HistoryItemModel(**{**i, "saved": bool(i["saved"])}) for i in saved],
        safety_alerts=safety_alerts[:6],
    )


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Novelty addition: lightweight feedback loop. Stores raw feedback for
    manual review, and — for 'not_helpful'/'helpful' ratings that reference
    specific RAG source ref_ids — nudges that document's retrieval trust
    score up/down (see app/database.py and app/rag/*.py). This reweights
    existing evidence; it does not retrain or learn new facts.
    """
    if req.rating not in ("helpful", "not_helpful"):
        raise HTTPException(status_code=400, detail="rating must be 'helpful' or 'not_helpful'.")
    save_feedback(
        query=req.query, rating=req.rating, query_type=req.query_type,
        session_id=req.session_id, comment=req.comment, source_ref_ids=req.source_ref_ids,
    )
    return {"status": "recorded"}


@app.get("/api/feedback/stats", response_model=FeedbackStats)
def feedback_stats():
    return FeedbackStats(**get_feedback_stats())


# Serve the full frontend (index.html, login.html, dashboard.html, css/, js/, ...)
# directly from the app's static directory. StaticFiles(html=True) auto-serves
# index.html for "/" and serves every other page/asset at its own path (e.g.
# /login.html, /css/styles.css) so the frontend's plain relative-path links
# ("css/styles.css", "dashboard.html", ...) resolve correctly. Mounted last so
# it never shadows the /api/* routes registered above.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
