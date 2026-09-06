"""
Phase 1: Eval set generator.

Builds a ~150-question evaluation set covering:
  - medicine_info questions (usage / side effects / warnings) for every
    seeded medicine — deterministic templates, so these always exist
    regardless of LLM availability.
  - drug_interaction questions for every seeded interaction pair
    (deterministic templates).
  - LLM-bootstrapped paraphrases of a sample of the above, to test
    robustness to phrasing (uses the same Gemini/Grok/mock LLM client
    as the app — falls back to the template question itself in mock
    mode, so the script still produces a full set with zero cost).

Output: eval/eval_set.json — a list of
  {"id", "question", "type", "expected_medicines", "expected_interaction"}

Run:
    python -m eval.generate_eval_set
"""
import json
import logging
import os

from app.database import (
    list_all_medicine_names, get_conn,
)
from app.llm_client import llm_client

logging.basicConfig(level="INFO")
logger = logging.getLogger("medagent.eval.generate")

OUT_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")

PARAPHRASE_SYSTEM_PROMPT = """Rewrite the following question in a different, natural way a real
patient might ask it, while keeping exactly the same medicines and meaning. Return ONLY the
rewritten question text, nothing else."""


def _paraphrase(question: str) -> str:
    try:
        result = llm_client.complete(PARAPHRASE_SYSTEM_PROMPT, question, json_mode=False, temperature=0.7)
        result = result.strip().strip('"')
        return result if result else question
    except Exception as e:
        logger.warning(f"Paraphrase failed for {question!r}: {e}")
        return question


def build_eval_set(include_paraphrases: bool = True) -> list[dict]:
    from app.database import init_db
    init_db()

    medicines = list_all_medicine_names()
    with get_conn() as conn:
        interactions = conn.execute("SELECT drug1, drug2, severity FROM drug_interactions").fetchall()

    cases = []
    case_id = 0

    # --- medicine_info questions (deterministic templates) ---
    templates = [
        "What is {med} used for?",
        "What are the side effects of {med}?",
        "What warnings should I know about {med}?",
        "Is {med} safe to take regularly?",
        "How does {med} work?",
        "What should I know before taking {med}?",
    ]
    for med in medicines:
        for t in templates:
            case_id += 1
            q = t.format(med=med)
            cases.append({
                "id": f"info_{case_id}",
                "question": q,
                "type": "medicine_info",
                "expected_medicines": [med],
                "expected_interaction": None,
            })

    # --- drug_interaction questions (deterministic templates) ---
    interaction_templates = [
        "Can {a} and {b} be taken together?",
        "Is there an interaction between {a} and {b}?",
        "I'm taking {a}, is it safe to also take {b}?",
        "Does {a} interact with {b}?",
        "What happens if I combine {a} and {b}?",
    ]
    for row in interactions:
        for t in interaction_templates:
            case_id += 1
            q = t.format(a=row["drug1"], b=row["drug2"])
            cases.append({
                "id": f"interaction_{case_id}",
                "question": q,
                "type": "drug_interaction",
                "expected_medicines": [row["drug1"], row["drug2"]],
                "expected_interaction": row["severity"],
            })

    # --- general / edge cases ---
    edge_cases = [
        ("Hello, how are you?", "general", [], None),
        ("Can Xanaxolol and Fooboprofen be taken together?", "drug_interaction", [], "unknown_medicine"),
        ("What is Metformin used for?", "medicine_info", ["Metformin"], None),
    ]
    for q, t, meds, exp in edge_cases:
        case_id += 1
        cases.append({
            "id": f"edge_{case_id}", "question": q, "type": t,
            "expected_medicines": meds, "expected_interaction": exp,
        })

    # --- LLM-bootstrapped paraphrases of a sample, for phrasing robustness ---
    if include_paraphrases:
        sample = cases[::3]  # every 3rd case
        for base in sample:
            case_id += 1
            paraphrased_q = _paraphrase(base["question"])
            cases.append({
                "id": f"paraphrase_{case_id}",
                "question": paraphrased_q,
                "type": base["type"],
                "expected_medicines": base["expected_medicines"],
                "expected_interaction": base["expected_interaction"],
                "paraphrase_of": base["id"],
            })

    return cases


def main():
    cases = build_eval_set()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    logger.info(f"Wrote {len(cases)} eval cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
