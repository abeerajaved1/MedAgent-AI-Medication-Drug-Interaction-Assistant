"""
Phase 1 upgrade: real vector database (Chroma) + medical-domain embeddings,
replacing the TF-IDF retriever.

Uses a sentence-transformers-compatible PubMedBERT model
("pritamdeka/S-PubMedBert-MS-MARCO") so retrieval understands medical
terminology/synonyms instead of just keyword overlap. Chroma runs
embedded (no server, no extra infra) and persists to disk, so this stays
100% free and Docker/free-tier friendly.

If the embedding model can't be downloaded (e.g. no internet at build
time, or a fully offline environment), this module logs a warning and
the RAG Agent falls back to the TF-IDF retriever automatically — the
app never hard-fails because of this upgrade.

Novelty addition — feedback-driven re-ranking: search() over-fetches
candidates and re-ranks them by (similarity_score * trust_multiplier),
where trust_multiplier comes from app.database.get_trust_multiplier()
and is nudged up/down by the /api/feedback endpoint. This is the
retrieval-side half of the "continual improvement without full
retraining" feedback loop described in the project proposal.
"""
import json
import logging
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

logger = logging.getLogger("medagent.chroma_store")

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base.json")
COLLECTION_NAME = "medagent_kb"
OVER_FETCH_MULTIPLIER = 3  # fetch this many x top_k candidates before trust re-ranking


@dataclass
class RetrievedDoc:
    id: str
    medicine: str
    title: str
    text: str
    score: float


class ChromaKnowledgeBase:
    """Same public interface as the old TF-IDF KnowledgeBase (search /
    search_by_medicine) so it's a drop-in replacement in rag_agent.py."""

    def __init__(self, kb_path: str = KB_PATH):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME
        )

        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )

        if self.collection.count() == 0:
            self._index_documents()

        logger.info(
            f"Chroma knowledge base ready: {self.collection.count()} documents "
            f"(embedding model: {settings.EMBEDDING_MODEL_NAME})."
        )

    def _index_documents(self):
        self.collection.add(
            ids=[d["id"] for d in self.docs],
            documents=[f"{d['title']}. {d['text']}" for d in self.docs],
            metadatas=[
                {"medicine": d["medicine"], "medicine_lower": d["medicine"].lower(), "title": d["title"]}
                for d in self.docs
            ],
        )
        logger.info(f"Indexed {len(self.docs)} documents into Chroma.")

    def search(self, query: str, top_k: int = 3, medicine_filter: str | None = None) -> list[RetrievedDoc]:
        from app.database import get_trust_multiplier

        where = {"medicine_lower": medicine_filter.lower()} if medicine_filter else None
        fetch_n = min(top_k * OVER_FETCH_MULTIPLIER, len(self.docs)) or top_k
        result = self.collection.query(query_texts=[query], n_results=fetch_n, where=where)

        candidates = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma returns a distance (lower = closer); convert to a
            # similarity-like score in [0, 1] for compatibility with the
            # old TF-IDF interface.
            similarity = max(0.0, 1.0 - dist)
            trust = get_trust_multiplier(doc_id)
            original = next((d for d in self.docs if d["id"] == doc_id), None)
            candidates.append(RetrievedDoc(
                id=doc_id, medicine=meta.get("medicine", ""), title=meta.get("title", ""),
                text=original["text"] if original else doc_text, score=round(similarity * trust, 4),
            ))

        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates[:top_k]

    def search_by_medicine(self, medicine_name: str) -> list[RetrievedDoc]:
        result = self.collection.get(where={"medicine_lower": medicine_name.lower()})
        docs = []
        for doc_id, meta in zip(result.get("ids", []), result.get("metadatas", [])):
            original = next((d for d in self.docs if d["id"] == doc_id), None)
            if original:
                docs.append(RetrievedDoc(
                    id=doc_id, medicine=original["medicine"], title=original["title"],
                    text=original["text"], score=1.0,
                ))
        return docs


def build_knowledge_base():
    """Returns a Chroma-backed KB, or falls back to the TF-IDF KB on failure."""
    try:
        return ChromaKnowledgeBase()
    except Exception as e:
        logger.warning(f"Chroma/embedding model unavailable ({e}); falling back to TF-IDF retrieval.")
        from app.rag.knowledge_base import knowledge_base as tfidf_kb
        return tfidf_kb


knowledge_base = build_knowledge_base()
