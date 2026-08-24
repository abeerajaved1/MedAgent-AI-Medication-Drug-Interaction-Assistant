"""
Lightweight, dependency-friendly RAG retriever.

Uses TF-IDF (scikit-learn) instead of neural embeddings so the whole
project stays small and fast enough to build/run on free-tier hosting
(e.g. 512MB RAM instances) without downloading large embedding models.
"""
import json
import logging
import os
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("medagent.rag")

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base.json")


@dataclass
class RetrievedDoc:
    id: str
    medicine: str
    title: str
    text: str
    score: float


class KnowledgeBase:
    def __init__(self, kb_path: str = KB_PATH):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self.corpus = [f"{d['medicine']} {d['title']} {d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)
        logger.info(f"RAG knowledge base loaded: {len(self.docs)} documents.")

    def search(self, query: str, top_k: int = 3, medicine_filter: str | None = None) -> list[RetrievedDoc]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()

        ranked_idx = sims.argsort()[::-1]
        results = []
        for idx in ranked_idx:
            if sims[idx] <= 0:
                continue
            doc = self.docs[idx]
            if medicine_filter and medicine_filter.lower() not in doc["medicine"].lower():
                continue
            results.append(RetrievedDoc(
                id=doc["id"], medicine=doc["medicine"], title=doc["title"],
                text=doc["text"], score=float(sims[idx]),
            ))
            if len(results) >= top_k:
                break
        return results

    def search_by_medicine(self, medicine_name: str) -> list[RetrievedDoc]:
        results = []
        for doc in self.docs:
            if doc["medicine"].lower() == medicine_name.lower():
                results.append(RetrievedDoc(
                    id=doc["id"], medicine=doc["medicine"], title=doc["title"],
                    text=doc["text"], score=1.0,
                ))
        return results


knowledge_base = KnowledgeBase()
