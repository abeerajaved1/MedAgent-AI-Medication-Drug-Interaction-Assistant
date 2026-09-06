"""
Brand-to-generic name resolution (new capability — extends the existing
RxNorm integration in app/agents/external_source_agent.py, which already
calls RxNorm purely for name validation, to also expose RxNorm's free
brand<->generic (ingredient) mapping).

Resolves e.g. "Panadol" -> "Paracetamol" (or the reverse), so the local
SQL database / interaction graph / RAG knowledge base — which are keyed
on a mix of generic and a few well-known brand names — can still be
matched against international brand names the demo dataset never
explicitly listed. This extends the project's WHO EML "global
relevance" narrative beyond US-labeled drug names.

Uses only RxNorm's free, no-API-key REST endpoints:
  https://rxnav.nlm.nih.gov/REST/rxcui.json?name=...
  https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN

Fails soft: any network error or unrecognized name returns None rather
than raising, so callers can always fall back to using the original
name unresolved.
"""
import logging

import httpx

from app.config import settings

logger = logging.getLogger("medagent.name_resolution")

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


class NameResolutionAgent:
    def __init__(self):
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
            logger.info(f"RxNorm rxcui lookup failed for '{name}': {e}")
            return None

    def resolve_generic(self, name: str) -> str | None:
        """Given a name that might be a brand name, returns the generic
        (active ingredient) name RxNorm associates with it, or None if it
        can't be resolved or the input is already the generic name."""
        if not settings.EXTERNAL_SOURCES_ENABLED:
            return None

        rxcui = self._get_rxcui(name)
        if not rxcui:
            return None

        try:
            r = self.client.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json", params={"tty": "IN"})
            r.raise_for_status()
            groups = r.json().get("relatedGroup", {}).get("conceptGroup", []) or []
            for group in groups:
                if group.get("tty") == "IN":
                    properties = group.get("conceptProperties", []) or []
                    if properties:
                        generic_name = properties[0].get("name")
                        if generic_name and generic_name.strip().lower() != name.strip().lower():
                            logger.info(f"Resolved brand/alias '{name}' -> generic '{generic_name}' via RxNorm.")
                            return generic_name
            return None
        except Exception as e:
            logger.info(f"RxNorm brand->generic lookup failed for '{name}' (rxcui={rxcui}): {e}")
            return None


name_resolution_agent = NameResolutionAgent()
