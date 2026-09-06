"""
Sentence-level citation attribution (new differentiator vs. generic RAG
chatbots, which typically cite at the document level only — "these 3
sources were used somewhere").

This module maps each sentence of the FINAL answer to the specific
evidence source(s) that best support it, producing footnote-style
[1] / [2] markers tied to exact source snippets. This turns the
"citation coverage / source attribution precision" evaluation metric
from a binary yes/no into something numerically measurable: what
fraction of sentences have a supported citation above a similarity
threshold, and how strong is that support.

Deliberately dependency-light and free: uses TF-IDF + cosine similarity
(scikit-learn, already a project dependency via app/rag/knowledge_base.py)
rather than a second LLM call, so this costs no extra API spend and adds
negligible latency.

Limitations, stated honestly for any write-up:
  - This is LEXICAL similarity, not semantic entailment. A sentence that
    paraphrases a source very differently in wording may score low even
    though it's faithful, and conversely a sentence that borrows
    distinctive vocabulary from an unrelated source can score higher
    than it should. It is a strong, cheap proxy — not a formal
    fact-verification system.
  - Sentences with no source above SIMILARITY_THRESHOLD are labeled
    "general/unsupported" rather than force-matched to the closest
    source, which is what makes the coverage metric meaningful.
"""
import logging
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("medagent.citation_attribution")

SIMILARITY_THRESHOLD = 0.12  # empirically low because sentences/snippets are short
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class EvidenceUnit:
    """A single citable piece of evidence, keyed to its display index."""
    index: int          # 1-based, matches the footnote marker shown to the user
    label: str          # short human-readable source name (SourceRef.source)
    text: str           # the actual evidence text to match sentences against


@dataclass
class SentenceCitation:
    sentence: str
    cited_indices: list[int] = field(default_factory=list)
    best_score: float = 0.0
    supported: bool = False


@dataclass
class AttributionResult:
    sentence_citations: list[SentenceCitation] = field(default_factory=list)
    annotated_text: str = ""
    coverage: float = 0.0   # fraction of substantive sentences with >=1 supported citation


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    raw = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def _is_substantive(sentence: str) -> bool:
    """Filters out boilerplate/disclaimer-style sentences that shouldn't
    count against citation coverage (they're not evidence claims)."""
    lowered = sentence.lower()
    boilerplate_markers = [
        "consult a", "speak to a", "talk to your", "not a substitute",
        "seek emergency", "call your local emergency",
    ]
    return len(sentence.split()) >= 4 and not any(m in lowered for m in boilerplate_markers)


def attribute_citations(answer_text: str, evidence_units: list[EvidenceUnit],
                         max_citations_per_sentence: int = 2) -> AttributionResult:
    sentences = _split_sentences(answer_text)
    if not sentences or not evidence_units:
        return AttributionResult(
            sentence_citations=[SentenceCitation(sentence=s) for s in sentences],
            annotated_text=answer_text,
            coverage=0.0,
        )

    corpus = [e.text for e in evidence_units] + sentences
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary (e.g. all evidence/sentences were stopwords) — no citations possible.
        return AttributionResult(
            sentence_citations=[SentenceCitation(sentence=s) for s in sentences],
            annotated_text=answer_text,
            coverage=0.0,
        )

    n_evidence = len(evidence_units)
    evidence_matrix = matrix[:n_evidence]
    sentence_matrix = matrix[n_evidence:]
    sims = cosine_similarity(sentence_matrix, evidence_matrix)  # shape (n_sentences, n_evidence)

    citations: list[SentenceCitation] = []
    annotated_parts: list[str] = []
    substantive_count = 0
    supported_count = 0

    for i, sentence in enumerate(sentences):
        scores = sims[i]
        ranked = sorted(range(n_evidence), key=lambda j: scores[j], reverse=True)
        top = [j for j in ranked if scores[j] >= SIMILARITY_THRESHOLD][:max_citations_per_sentence]

        best_score = float(scores[ranked[0]]) if ranked else 0.0
        cited_indices = [evidence_units[j].index for j in top]
        supported = bool(cited_indices)

        citations.append(SentenceCitation(
            sentence=sentence, cited_indices=cited_indices,
            best_score=round(best_score, 4), supported=supported,
        ))

        marker = "".join(f"[{idx}]" for idx in cited_indices)
        annotated_parts.append(f"{sentence}{(' ' + marker) if marker else ''}")

        if _is_substantive(sentence):
            substantive_count += 1
            if supported:
                supported_count += 1

    coverage = round(supported_count / substantive_count, 3) if substantive_count else 0.0

    return AttributionResult(
        sentence_citations=citations,
        annotated_text=" ".join(annotated_parts),
        coverage=coverage,
    )


def build_evidence_units(sources: list) -> list[EvidenceUnit]:
    """Builds EvidenceUnit list from a pipeline's list of SourceRef-like
    objects (must expose .source and .snippet attributes), assigning
    stable 1-based indices in list order for footnote display."""
    units = []
    for i, s in enumerate(sources, start=1):
        text = getattr(s, "snippet", "") or ""
        label = getattr(s, "source", f"Source {i}")
        if text:
            units.append(EvidenceUnit(index=i, label=label, text=text))
    return units
