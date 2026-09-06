"""
Self-consistency safety check (new capability — a known, citable
technique for hallucination/uncertainty detection: draft the answer
twice with different sampling and treat disagreement between the two as
an additional uncertainty signal, on top of the Safety Agent's own
verification).

Cost: exactly one extra LLM call (the second draft). The comparison
itself is done with cheap lexical similarity (TF-IDF cosine, no extra
LLM call) rather than a third LLM call, to keep this feature close to
free while still being a real, working signal — not a token gesture.

This does NOT replace the Safety/Verifier Agent. It produces one more
input (`flagged: bool`) that the Safety Agent / Escalation Agent can use
to be more conservative, exactly the way real self-consistency
hallucination-detection papers use multiple-sample agreement as a proxy
for confidence.
"""
import logging
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("medagent.consistency_check")

DISAGREEMENT_THRESHOLD = 0.55  # cosine similarity below this => flagged as disagreement


@dataclass
class ConsistencyResult:
    flagged: bool
    similarity: float
    draft_a: str
    draft_b: str


def _lexical_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([a, b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except ValueError:
        return 1.0  # degenerate case (e.g. both empty/stopword-only) — don't false-flag


class SelfConsistencyChecker:
    def check(self, pharmacist_agent, user_query: str, evidence_text: str,
              patient_profile: dict | None = None) -> ConsistencyResult:
        """
        `pharmacist_agent` is passed in (rather than imported) to avoid a
        circular import between pharmacist_agent.py and this module, and
        so the eval harness can inject a stub for offline testing.
        """
        draft_a = pharmacist_agent.draft_response(
            user_query, evidence_text, patient_profile=patient_profile, temperature=0.2,
        )
        draft_b = pharmacist_agent.draft_response(
            user_query, evidence_text, patient_profile=patient_profile, temperature=0.9,
        )

        similarity = _lexical_similarity(draft_a, draft_b)
        flagged = similarity < DISAGREEMENT_THRESHOLD
        if flagged:
            logger.info(f"Self-consistency check flagged disagreement (similarity={similarity:.3f}).")

        return ConsistencyResult(flagged=flagged, similarity=round(similarity, 3),
                                  draft_a=draft_a, draft_b=draft_b)


self_consistency_checker = SelfConsistencyChecker()
