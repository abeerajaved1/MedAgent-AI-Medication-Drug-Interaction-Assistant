"""
Phase 1: Eval harness.

Runs every question in eval/eval_set.json through the real MedAgent
pipeline (app/pipeline.py — same code path as the API) and computes:

  - retrieval_hit_rate      fraction of questions that returned >=1 source
  - avg_sources_per_query   citation coverage
  - verification_status distribution (approved / warning / insufficient_evidence)
  - avg_revision_count      how often the Phase-3 veto loop fired
  - groundedness_rate       LLM-judge check: is the explanation supported
                             by the evidence that was actually retrieved?
                             (a lightweight hallucination-rate proxy)

Run:
    python -m eval.generate_eval_set   # first time, or to regenerate
    python -m eval.run_eval
"""
import json
import logging
import os
from collections import Counter

from app.pipeline import run_medagent_pipeline
from app.llm_client import llm_client

logging.basicConfig(level="WARNING")  # quiet the agent pipeline's own INFO logs during eval
logger = logging.getLogger("medagent.eval.run")

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge. Given evidence and a generated answer,
decide if the answer is fully supported by the evidence (no invented facts).
Return STRICT JSON only: {"grounded": true|false, "reason": "one short sentence"}"""


def _judge_grounded(evidence_text: str, explanation: str) -> bool | None:
    try:
        raw = llm_client.complete(
            JUDGE_SYSTEM_PROMPT,
            f"--- Evidence ---\n{evidence_text}\n--- Answer ---\n{explanation}",
            json_mode=True,
        )
        data = json.loads(raw)
        return bool(data.get("grounded"))
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return None


def run_eval(limit: int | None = None, judge_sample_rate: int = 3,
             session_id: str | None = None, verbose: bool = True,
             write_results: bool = True):
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if limit:
        cases = cases[:limit]

    results = []
    status_counts = Counter()
    total_sources = 0
    hit_count = 0
    total_revisions = 0
    grounded_checks = []

    for i, case in enumerate(cases):
        response, debug = run_medagent_pipeline(case["question"], session_id=session_id, return_debug=True)

        n_sources = len(response.sources)
        total_sources += n_sources
        if n_sources > 0:
            hit_count += 1
        status_counts[response.verification_status] += 1
        total_revisions += response.revision_count

        grounded = None
        # Judge only a sample (every Nth case) to keep eval cost/time bounded.
        if i % judge_sample_rate == 0 and debug and debug.evidence_text:
            grounded = _judge_grounded(debug.evidence_text, response.explanation)
            if grounded is not None:
                grounded_checks.append(grounded)

        results.append({
            "id": case["id"],
            "question": case["question"],
            "expected_type": case["type"],
            "actual_type": response.query_type,
            "type_match": case["type"] == response.query_type,
            "n_sources": n_sources,
            "verification_status": response.verification_status,
            "revision_count": response.revision_count,
            "grounded": grounded,
        })

    n = len(cases)
    summary = {
        "n_cases": n,
        "retrieval_hit_rate": round(hit_count / n, 3) if n else None,
        "avg_sources_per_query": round(total_sources / n, 2) if n else None,
        "avg_revision_count": round(total_revisions / n, 2) if n else None,
        "verification_status_distribution": dict(status_counts),
        "intent_classification_accuracy": round(
            sum(1 for r in results if r["type_match"]) / n, 3
        ) if n else None,
        "groundedness_rate_sampled": round(sum(grounded_checks) / len(grounded_checks), 3)
        if grounded_checks else None,
        "groundedness_sample_size": len(grounded_checks),
    }

    if write_results:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)

    if verbose:
        print(json.dumps(summary, indent=2))
        print(f"\nFull per-question results written to {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    run_eval()
