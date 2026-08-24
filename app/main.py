"""
MedAgent — main FastAPI application.

Wires together the full workflow described in the project spec:

User Query -> Coordinator Agent -> Understand Intent -> Select Tools ->
RAG / SQL / Drug Interaction Tool -> Combine Retrieved Information ->
Pharmacist/Information Agent (Draft) -> Safety/Verifier Agent ->
Approved / Warning / Insufficient Evidence -> Final Response
"""
import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.models import ChatRequest, ChatResponse, SourceRef
from app.agents.coordinator import coordinator_agent
from app.agents.rag_agent import rag_agent
from app.agents.sql_agent import sql_agent
from app.agents.interaction_tool import drug_interaction_tool
from app.agents.pharmacist_agent import pharmacist_agent
from app.agents.safety_agent import safety_verifier_agent
from app.agents.external_source_agent import external_source_agent

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("medagent.main")

app = FastAPI(title="MedAgent", version="1.0.0",
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
    logger.info(f"MedAgent started. LLM provider = {settings.LLM_PROVIDER}")


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()

    # 1. Coordinator: understand intent + select tools
    decision = coordinator_agent.route(query)

    sources: list[SourceRef] = []
    evidence_parts: list[str] = []
    interaction_status = None
    severity = None
    has_warnings = False
    evidence_found = True

    # 2. Tool execution based on routing decision
    if "rag" in decision.required_tools:
        rag_result = rag_agent.retrieve(query, decision.medicines)
        if rag_result.documents:
            evidence_parts.append("RAG knowledge base results:\n" + rag_result.as_context_text())
            for d in rag_result.documents:
                sources.append(SourceRef(source=d.title, snippet=d.text[:180]))
        else:
            evidence_found = False

    if "sql" in decision.required_tools:
        for med in decision.medicines:
            sql_result = sql_agent.lookup_medicine(med)
            if sql_result.found:
                evidence_parts.append(
                    f"SQL database record for {med}: {sql_result.medicine_info}"
                )
                sources.append(SourceRef(source=f"Database record: {med}",
                                          snippet=str(sql_result.medicine_info)))
                if sql_result.warnings:
                    has_warnings = True
                    for w in sql_result.warnings:
                        evidence_parts.append(f"Safety warning for {med}: {w['description']}")
                        sources.append(SourceRef(source=f"Safety warning: {w['warning_type']}",
                                                  snippet=w["description"][:180]))

    if "interaction_tool" in decision.required_tools and len(decision.medicines) >= 2:
        check = drug_interaction_tool.check(decision.medicines[0], decision.medicines[1])
        if check.status == "found":
            interaction_status = check.severity
            severity = check.severity
            evidence_parts.append(
                f"Drug interaction found between {check.normalized_a} and "
                f"{check.normalized_b} (severity: {check.severity}): {check.description}"
            )
            sources.append(SourceRef(
                source=f"Drug Interaction DB: {check.normalized_a} + {check.normalized_b}",
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

    if "external" in decision.required_tools and decision.medicines:
        # External Source Agent: live lookups from RxNorm + openFDA so the
        # system isn't limited to the local RAG/DB data.
        for med in decision.medicines:
            ext_result = external_source_agent.lookup_medicine(med)
            if ext_result.found:
                evidence_found = True
                for d in ext_result.docs:
                    evidence_parts.append(f"{d.source_name}: {d.text}")
                    sources.append(SourceRef(source=d.source_name, snippet=d.text[:180]))

        if "interaction_tool" in decision.required_tools and len(decision.medicines) >= 2:
            ext_hint = external_source_agent.check_interaction_hint(
                decision.medicines[0], decision.medicines[1]
            )
            if ext_hint:
                evidence_parts.append(f"{ext_hint.source_name}: {ext_hint.text}")
                sources.append(SourceRef(source=ext_hint.source_name, snippet=ext_hint.text[:180]))

    if decision.intent == "general" or not evidence_parts:
        evidence_parts.append(
            "No specific medication evidence was retrieved for this query. "
            "It may be a general question, greeting, or a medicine not in the "
            "current knowledge base."
        )
        if decision.intent == "general":
            evidence_found = True  # not a failure, just not a medication query
        else:
            evidence_found = False

    evidence_text = "\n\n".join(evidence_parts)

    # 3. Pharmacist/Information Agent: draft response grounded in evidence
    draft = pharmacist_agent.draft_response(query, evidence_text)

    # 4. Safety/Verifier Agent: final validation layer
    verification = safety_verifier_agent.verify(
        user_query=query,
        evidence_text=evidence_text,
        draft_response=draft,
        severity=severity,
        has_warnings=has_warnings,
        evidence_found=evidence_found,
    )

    # 5. Assemble final structured response
    if verification.status == "insufficient_evidence":
        explanation = (
            "I don't have enough reliable information in my available data to answer "
            "this confidently. " + draft
        )
    else:
        explanation = draft

    return ChatResponse(
        query_type=decision.intent,
        medicines=decision.medicines,
        interaction_status=interaction_status,
        explanation=explanation,
        safety_note=verification.safety_note,
        verification_status=verification.status,
        sources=sources,
    )


# Serve the simple chat UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
