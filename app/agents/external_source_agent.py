"""
External Source Agent (extends the original 6-agent spec).

Queries free, no-API-key-required public data sources so the system is
not limited to the local RAG knowledge base / seed database:

  - RxNorm (U.S. National Library of Medicine) — normalizes / validates
    drug names against a real, live medical terminology database.
    https://rxnav.nlm.nih.gov/REST  (no key required)

  - openFDA Drug Label API (U.S. FDA) — pulls indications, warnings,
    adverse reactions, contraindications, and (when present) the
    manufacturer's own drug-interactions section straight from the
    official product label.
    https://api.fda.gov/drug/label.json  (no key required for normal use)

Every call is short-timeout and wrapped so that a network failure or an
unrecognized drug degrades gracefully to "not found" rather than
breaking the pipeline — the rest of the system (local RAG + SQL data)
still works even if these external services are unreachable.
"""
import logging
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger("medagent.external_source")

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

# Label fields worth surfacing, in priority order, with a display title
# and a hard character cap per field (labels can be very long).
LABEL_FIELDS = [
    ("indications_and_usage", "Indications & Usage (FDA label)", 600),
    ("warnings", "Warnings (FDA label)", 600),
    ("warnings_and_cautions", "Warnings & Cautions (FDA label)", 600),
    ("contraindications", "Contraindications (FDA label)", 500),
    ("adverse_reactions", "Adverse Reactions (FDA label)", 500),
    ("drug_interactions", "Drug Interactions (FDA label)", 600),
]


@dataclass
class ExternalDoc:
    source_name: str
    url: str
    text: str


@dataclass
class ExternalLookupResult:
    medicine: str
    found: bool
    rxcui: str | None = None
    docs: list[ExternalDoc] = field(default_factory=list)


class ExternalSourceAgent:
    def __init__(self):
        self.enabled = settings.EXTERNAL_SOURCES_ENABLED
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=settings.EXTERNAL_TIMEOUT_SECONDS)
        return self._client

    def _get_rxcui(self, name: str) -> str | None:
        try:
            r = self.client.get(f"{RXNORM_BASE}/rxcui.json", params={"name": name})
            r.raise_for_status()
            ids = r.json().get("idGroup", {}).get("rxnormId")
            return ids[0] if ids else None
        except Exception as e:
            logger.info(f"RxNorm lookup unavailable for '{name}': {e}")
            return None

    def _get_openfda_label(self, name: str) -> dict | None:
        try:
            query = f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"'
            r = self.client.get(OPENFDA_BASE, params={"search": query, "limit": 1})
            r.raise_for_status()
            results = r.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            logger.info(f"openFDA lookup unavailable for '{name}': {e}")
            return None

    def lookup_medicine(self, name: str) -> ExternalLookupResult:
        if not self.enabled:
            return ExternalLookupResult(medicine=name, found=False)

        rxcui = self._get_rxcui(name)
        label = self._get_openfda_label(name)

        docs: list[ExternalDoc] = []
        if label:
            for field_key, title, cap in LABEL_FIELDS:
                values = label.get(field_key)
                if values:
                    text = " ".join(values).strip()[:cap]
                    docs.append(ExternalDoc(
                        source_name=title,
                        url="https://api.fda.gov/drug/label.json",
                        text=text,
                    ))

        found = bool(rxcui or docs)
        logger.info(f"External lookup for '{name}': rxcui={rxcui}, label_fields={len(docs)}")
        return ExternalLookupResult(medicine=name, found=found, rxcui=rxcui, docs=docs)

    def check_interaction_hint(self, drug_a: str, drug_b: str) -> ExternalDoc | None:
        """
        Best-effort cross-reference: checks each drug's own FDA label
        'drug_interactions' section for a mention of the other drug.
        This supplements (never replaces) the structured local
        interaction database and the Safety Agent's verification.
        """
        if not self.enabled:
            return None

        hints = []
        for primary, other in ((drug_a, drug_b), (drug_b, drug_a)):
            label = self._get_openfda_label(primary)
            if not label:
                continue
            text = " ".join(label.get("drug_interactions", []) or [])
            if other.lower() in text.lower():
                hints.append(f"{primary} label mentions {other}: {text[:400]}")

        if not hints:
            return None
        return ExternalDoc(
            source_name=f"openFDA label cross-reference: {drug_a} + {drug_b}",
            url="https://api.fda.gov/drug/label.json",
            text=" | ".join(hints),
        )


external_source_agent = ExternalSourceAgent()
