"""
Full reasoning-trace export (new differentiator — "transparency by
design"). Persists a step-by-step record of which agents fired, what
each contributed, and every safety veto/revision round for a single
query, retrievable afterwards via GET /api/trace/{query_id}.

This is the concrete evidence for a safety-critical AI paper's
regulatory/ethics section that the system is not a black box: every
answer can be traced back to exactly which tools ran, what evidence
they returned, and why the Safety Agent approved, warned, or vetoed.

Storage: a dedicated `reasoning_traces` SQLite table (see the
REPLACE_FILES/app/database.py additions — `save_trace` / `get_trace`).
The trace itself is stored as a JSON blob; this module only builds and
serializes it, it does not know about SQLite directly, so it can be
reused by the eval harness (eval/run_eval.py) without a live DB if
needed.
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("medagent.reasoning_trace")


@dataclass
class TraceStep:
    agent: str
    action: str
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningTrace:
    query_id: str
    session_id: str | None
    query: str
    steps: list[TraceStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def log(self, agent: str, action: str, detail: str = ""):
        # Truncate detail defensively — traces are meant to be inspectable,
        # not a full evidence dump; the evidence itself is already visible
        # in the ChatResponse.sources field.
        self.steps.append(TraceStep(agent=agent, action=action, detail=detail[:800]))
        logger.debug(f"[trace {self.query_id}] {agent}: {action} — {detail[:120]}")

    def finalize(self) -> dict:
        self.finished_at = time.time()
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.finished_at - self.started_at, 3),
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.finalize())


def new_trace(query: str, session_id: str | None = None) -> ReasoningTrace:
    return ReasoningTrace(query_id=str(uuid.uuid4()), session_id=session_id, query=query)
