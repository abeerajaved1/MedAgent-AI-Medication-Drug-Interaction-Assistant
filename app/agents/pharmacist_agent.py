"""
3.5 Pharmacist / Information Agent

Creates a clear, structured DRAFT response using only the information
retrieved from the RAG knowledge base, the SQL database, and the Drug
Interaction Tool. It is explicitly instructed not to add facts beyond the
supplied evidence.
"""
import logging

from app.llm_client import llm_client

logger = logging.getLogger("medagent.pharmacist_agent")

DRAFT_SYSTEM_PROMPT = """You are the Pharmacist/Information Agent in a medication information system.
Write a clear, plain-language draft answer for the user using ONLY the evidence
provided below. Do not invent facts, dosages, or claims that are not present in
the evidence. If the evidence is insufficient to answer part of the question,
say so explicitly instead of guessing.

Keep the tone calm, factual, and non-alarming. Do not give a definitive medical
recommendation — describe what the evidence shows and note that a doctor or
pharmacist should be consulted for personal medical decisions.
"""


class PharmacistAgent:
    def draft_response(self, user_query: str, evidence_text: str) -> str:
        user_prompt = (
            f"User question: {user_query}\n\n"
            f"--- Evidence available ---\n{evidence_text}\n"
            f"--- End evidence ---\n\n"
            "Write the draft response now."
        )
        draft = llm_client.complete(DRAFT_SYSTEM_PROMPT, user_prompt, json_mode=False, temperature=0.3)
        logger.info("Draft response generated.")
        return draft.strip()


pharmacist_agent = PharmacistAgent()
