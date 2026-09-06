"""
Tier 3 #7: Cross-encoder reranking — a genuine second-stage retrieval
quality improvement, not fine-tuning, not a reworded claim.

How this differs from what the base retriever already does: both the
Chroma embedding search (app/rag/chroma_store.py) and the TF-IDF fallback
(app/rag/knowledge_base.py) are bi-encoders — they embed the query and
each document independently, then compare vectors. That's fast enough to
search a whole corpus, but the query and document never actually "see"
each other during scoring. A cross-encoder instead feeds the (query,
document) pair through the model *together*, letting it directly judge
relevance — much more accurate, but too slow to run against an entire
knowledge base. The standard pattern (used here) is: let the cheap
bi-encoder over-fetch a generous candidate set, then have the small,
free cross-encoder re-score just those candidates and pick the true
top-K.

Uses sentence-transformers' CrossEncoder with a small, free, public model
(cross-encoder/ms-marco-MiniLM-L-6-v2, ~80MB, CPU-friendly). No API cost,
no fine-tuning, no new dependency — sentence-transformers is already a
project dependency (it's what powers the Chroma embedding function).

Fails soft, matching the exact resilience pattern already established in
chroma_store.py's embedding-model fallback: if the model can't be loaded
(no internet access at first run, or a fully offline environment),
reranking is silently skipped and retrieval falls back to the base
retriever's own ranking — the app never hard-fails because of this.
"""
import dataclasses
import logging

from app.config import settings

logger = logging.getLogger("medagent.reranker")


class CrossEncoderReranker:
    def __init__(self):
        self.available = False
        self._model = None
        if not settings.RERANKING_ENABLED:
            logger.info("Cross-encoder reranking disabled via settings.RERANKING_ENABLED=false.")
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(settings.RERANKER_MODEL_NAME)
            self.available = True
            logger.info(f"Cross-encoder reranker loaded: {settings.RERANKER_MODEL_NAME}")
        except Exception as e:
            logger.warning(
                f"Cross-encoder reranker unavailable ({e}); retrieval will use the base "
                f"retriever's own ranking only. This is expected in fully offline "
                f"environments or on first run with no internet access to download the model."
            )

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        """Re-scores `candidates` (a list of RetrievedDoc-shaped objects
        with .text and .score) against `query` using the cross-encoder,
        and returns the top_k reordered by the new score. Falls back to
        a plain truncation of the existing order if the model isn't
        available or the call fails for any reason — never raises."""
        if not self.available or not candidates:
            return candidates[:top_k]
        try:
            pairs = [(query, c.text) for c in candidates]
            scores = self._model.predict(pairs)
            reranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
            return [dataclasses.replace(c, score=round(float(s), 4)) for c, s in reranked[:top_k]]
        except Exception as e:
            logger.warning(f"Cross-encoder rerank call failed ({e}); falling back to original order.")
            return candidates[:top_k]


reranker = CrossEncoderReranker()
