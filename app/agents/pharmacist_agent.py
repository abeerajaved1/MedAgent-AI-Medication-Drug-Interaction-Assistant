"""
3.5 Pharmacist / Information Agent

Creates a clear, structured DRAFT response using only the information
retrieved from the RAG knowledge base, the SQL database, and the Drug
Interaction Tool. It is explicitly instructed not to add facts beyond the
supplied evidence.

Phase 3 upgrade: draft_response() now accepts optional `safety_feedback`
from the Safety/Verifier Agent so the Pharmacist Agent can revise a
rejected draft instead of the system just tacking a warning onto it.

New in this revision: draft_response() accepts an optional `temperature`
override so the new Self-Consistency Checker (app/agents/consistency_check.py)
can request two independently sampled drafts of the same evidence to
compare for disagreement, without needing its own copy of this prompt.

Novelty addition — multilingual output: the draft is instructed to answer
in the same language the question was asked in. This is generation-side
only (translation happens when the LLM writes the final answer); the
underlying knowledge base and retrieval are English-only, so retrieval
quality for non-English queries depends on the LLM's cross-lingual
understanding of the query, not true multilingual retrieval. Document
this distinction honestly in any write-up.
"""
import logging

from app.llm_client import llm_client
from app.agents.reasoning_profile import REASONING_CHAIN_TEMPLATE, build_patient_context_block

logger = logging.getLogger("medagent.pharmacist_agent")

MULTILINGUAL_INSTRUCTION = """
LANGUAGE: Respond in the SAME language the user's question was written in,
even though the evidence provided to you is in English — translate and
synthesize naturally, don't answer in English if the question wasn't in
English. If the question mixes languages or the language is unclear,
default to English.
"""

DRAFT_SYSTEM_PROMPT = """You are the Pharmacist/Information Agent in a medication information system.
Write a clear, plain-language draft answer for the user using ONLY the evidence
provided below. Do not invent facts, dosages, or claims that are not present in
the evidence. If the evidence is insufficient to answer part of the question,
say so explicitly instead of guessing.

Keep the tone calm, factual, and non-alarming. Do not give a definitive medical
recommendation — describe what the evidence shows and note that a doctor or
pharmacist should be consulted for personal medical decisions.
""" + REASONING_CHAIN_TEMPLATE + MULTILINGUAL_INSTRUCTION

REVISION_SYSTEM_PROMPT = """You are the Pharmacist/Information Agent. Your previous draft was
REJECTED by the Safety/Verifier Agent. Revise the draft to directly address the
safety objection below while still grounding every claim in the evidence provided.
Do not ignore or soften the objection — it must be visibly reflected in the revised
response (e.g. an explicit warning, caveat, or a statement that the question cannot
be safely answered without professional input).
""" + REASONING_CHAIN_TEMPLATE + MULTILINGUAL_INSTRUCTION


class PharmacistAgent:
    def draft_response(self, user_query: str, evidence_text: str,
                        safety_feedback: str | None = None,
                        patient_profile: dict | None = None,
                        temperature: float = 0.3) -> str:
        context_block = build_patient_context_block(patient_profile)
        evidence_with_context = f"{context_block}\n\n{evidence_text}" if context_block else evidence_text

        if safety_feedback:
            user_prompt = (
                f"User question: {user_query}\n\n"
                f"--- Evidence available ---\n{evidence_with_context}\n"
                f"--- End evidence ---\n\n"
                f"--- Safety objection to address ---\n{safety_feedback}\n"
                f"--- End objection ---\n\n"
                "Write the revised draft response now."
            )
            system_prompt = REVISION_SYSTEM_PROMPT
        else:
            user_prompt = (
                f"User question: {user_query}\n\n"
                f"--- Evidence available ---\n{evidence_with_context}\n"
                f"--- End evidence ---\n\n"
                "Write the draft response now."
            )
            system_prompt = DRAFT_SYSTEM_PROMPT

        draft = llm_client.complete(system_prompt, user_prompt, json_mode=False, temperature=temperature)
        logger.info(f"{'Revised' if safety_feedback else 'Draft'} response generated "
                    f"(temperature={temperature}).")
        return draft.strip()


pharmacist_agent = PharmacistAgent()
