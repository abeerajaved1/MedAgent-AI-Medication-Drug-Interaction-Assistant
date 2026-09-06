"""
Multi-turn conversational memory (new capability — the blueprint claims
"Coordinator manages conversation state", which wasn't implemented; each
/api/chat call was previously stateless except for the profile/findings
tables).

Adds a real, session-scoped turn history so a follow-up like "what about
with my other medication?" or "is that safe for a child?" can be
resolved using the previous turn's context instead of being treated as
an isolated, ambiguous query.

Storage: a dedicated `conversation_turns` SQLite table (see
REPLACE_FILES/app/database.py additions — `save_turn` / `get_recent_turns`).
This module is a thin, DB-backed helper the Coordinator and Pharmacist
Agent call directly; it does not itself decide anything.
"""
import logging

from app import database as db

logger = logging.getLogger("medagent.conversation_memory")

DEFAULT_HISTORY_TURNS = 4  # user+assistant pairs to include as context


def save_turn(session_id: str, role: str, content: str) -> None:
    if not session_id:
        return
    db.save_conversation_turn(session_id, role, content)


def get_recent_turns(session_id: str, limit: int = DEFAULT_HISTORY_TURNS) -> list[dict]:
    if not session_id:
        return []
    return db.get_recent_conversation_turns(session_id, limit=limit)


def build_context_block(session_id: str | None, limit: int = DEFAULT_HISTORY_TURNS) -> str:
    """Formats recent turns into a short text block for injection into the
    Coordinator's classification prompt and/or the Pharmacist Agent's
    evidence context, so pronouns and implicit references ("my other
    medication", "that drug", "the one you just mentioned") can resolve
    against real prior turns instead of being guessed at."""
    if not session_id:
        return ""
    turns = get_recent_turns(session_id, limit=limit)
    if not turns:
        return ""
    lines = ["Recent conversation history (most recent last, for resolving references only — "
             "do not treat this as new evidence):"]
    for t in turns:
        speaker = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {t['content']}")
    return "\n".join(lines)
