"""
3.2 Medication Information and RAG Agent

Searches the medication knowledge base (TF-IDF retrieval) and returns
relevant evidence documents to ground the final response, reducing
reliance on the LLM's own (unverified) internal knowledge.
"""
import logging
from dataclasses import dataclass

from app.rag.knowledge_base import knowledge_base, RetrievedDoc
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
            docs = knowledge_base.search(query, top_k=settings.RAG_TOP_K)

        logger.info(f"RAG retrieved {len(docs)} documents for query: {query!r}")
        return RAGResult(documents=docs)


rag_agent = RAGAgent()
