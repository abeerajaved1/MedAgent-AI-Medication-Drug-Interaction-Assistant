"""
Phase 6: Ablation study harness (SO7 — "quantify the marginal
contribution of each agent/component").

Runs the same eval set under different configurations and reports how
each component changes the metrics, by toggling settings at runtime:

  - baseline        : local RAG + SQL + interaction graph only
  - +external        baseline + External Source Agent (RxNorm/openFDA)
  - +veto            +external + Safety Agent veto/re-draft loop
  - +profile (full)  +veto + Patient Profile Agent (allergy/polypharmacy),
                      run against a seeded demo profile so those checks
                      actually have something to fire on

Each configuration reuses eval/run_eval.py's metrics so results are
directly comparable across rows.

Run:
    python -m eval.generate_eval_set   # if not already generated
    python -m eval.run_ablation
"""
import json
import logging
import os

from app.config import settings
from app.database import init_db, upsert_patient_profile
from eval.run_eval import run_eval, EVAL_SET_PATH

logging.basicConfig(level="WARNING")
logger = logging.getLogger("medagent.eval.ablation")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ablation_results.json")
DEMO_SESSION_ID = "ablation-demo-session"


def _seed_demo_profile():
    """A fixed, illustrative profile so the +profile configuration's
    allergy/polypharmacy checks have real matches to fire on across the
    eval set's medicine_info/drug_interaction questions."""
    upsert_patient_profile(
        session_id=DEMO_SESSION_ID,
        age=52,
        allergies=["penicillin"],
        conditions=["hypertension"],
        current_medications=["Warfarin"],
    )


def run_ablation():
    if not os.path.exists(EVAL_SET_PATH):
        raise SystemExit("eval/eval_set.json not found — run `python -m eval.generate_eval_set` first.")

    init_db()
    _seed_demo_profile()

    configs = [
        {"name": "baseline (RAG+SQL+graph only)", "external": False, "veto_rounds": 0, "session_id": None},
        {"name": "+ external sources (RxNorm/openFDA)", "external": True, "veto_rounds": 0, "session_id": None},
        {"name": "+ safety veto/re-draft loop", "external": True, "veto_rounds": 1, "session_id": None},
        {"name": "+ patient profile (full system)", "external": True, "veto_rounds": 1, "session_id": DEMO_SESSION_ID},
    ]

    import sys
    limit = None
    if "--quick" in sys.argv:
        limit = 20
    for cfg in configs:
        cfg["limit"] = limit

    all_results = []
    for cfg in configs:
        settings.EXTERNAL_SOURCES_ENABLED = cfg["external"]
        settings.MAX_VETO_ROUNDS = cfg["veto_rounds"]

        logger.warning(f"\n=== Running ablation config: {cfg['name']} ===")
        summary = run_eval(session_id=cfg["session_id"], verbose=False, write_results=False,
                            limit=cfg.get("limit"))
        all_results.append({"config": cfg["name"], "settings": cfg, "summary": summary})

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== ABLATION SUMMARY ===")
    header = f"{'Config':40s} {'HitRate':>8s} {'AvgSrc':>8s} {'AvgRev':>8s} {'Warn%':>8s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        s = r["summary"]
        total = s["n_cases"]
        warn_pct = round(100 * s["verification_status_distribution"].get("warning", 0) / total, 1) if total else 0
        print(f"{r['config']:40s} {s['retrieval_hit_rate']:>8} {s['avg_sources_per_query']:>8} "
              f"{s['avg_revision_count']:>8} {warn_pct:>7}%")

    print(f"\nFull results written to {RESULTS_PATH}")
    return all_results


if __name__ == "__main__":
    run_ablation()
