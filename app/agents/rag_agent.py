"""
3.2 Medication Information and RAG Agent

Searches the medication knowledge base and returns relevant evidence
documents to ground the final response, reducing reliance on the LLM's
own (unverified) internal knowledge.

Tier 3 upgrade: when a plain-text semantic search is used (no exact
medicine-name filter), this over-fetches a generous candidate set from
the base retriever (TF-IDF or Chroma — whichever is active) and, if the
cross-encoder reranker is available (see app/rag/reranker.py), re-scores
those candidates for a more accurate final top-K. Falls back to exactly
the previous behavior (base retriever's own top-K, no over-fetch) when
reranking is disabled or unavailable, so this never changes retrieval
quality for the worse.

Medicine-filtered lookups (search_by_medicine) are unaffected — they're
an exact filter, not a similarity search, so there's nothing to rerank.
"""
import logging
from dataclasses import dataclass

from app.rag.chroma_store import knowledge_base, RetrievedDoc
from app.rag.reranker import reranker
from app.config import settings

logger = logging.getLogger("medagent.rag_agent")


@dataclass
class RAGResult:
    documents: list[RetrievedDoc]

    def as_context_text(self) -> str:
        if not self.documents:
            return "No relevant documents found in the knowledge base."
        chunks = []
        for d in self.documents:
            chunks.append(f"[{d.title}] (medicine: {d.medicine})\n{d.text}")
        return "\n\n".join(chunks)


class RAGAgent:
    def retrieve(self, query: str, medicines: list[str] | None = None) -> RAGResult:
        docs: list[RetrievedDoc] = []

        if medicines:
            for med in medicines:
                docs.extend(knowledge_base.search_by_medicine(med))

        if not docs:
            top_k = settings.RAG_TOP_K
            use_reranking = settings.RERANKING_ENABLED and reranker.available
            fetch_n = top_k * settings.RERANK_OVER_FETCH_MULTIPLIER if use_reranking else top_k

            candidates = knowledge_base.search(query, top_k=fetch_n)
            if use_reranking and len(candidates) > top_k:
                docs = reranker.rerank(query, candidates, top_k=top_k)
                logger.info(f"RAG reranked {len(candidates)} candidate(s) down to top {top_k}.")
            else:
                docs = candidates[:top_k]

        logger.info(f"RAG retrieved {len(docs)} documents for query: {query!r}")
        return RAGResult(documents=docs)


rag_agent = RAGAgent()
