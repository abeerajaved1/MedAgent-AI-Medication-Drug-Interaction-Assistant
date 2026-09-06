"""
Phase 4: Vision/Diagnostic Agent — preliminary chest X-ray analysis.

Free approach: instead of training/fine-tuning a vision-language model
(LLaVA-Med, Rad-DINO, etc. — needs GPU-hours and labeled radiology data),
this sends the image directly to Gemini's multimodal endpoint (free tier,
same GEMINI_API_KEY already used for text) with a tightly scoped prompt
that asks only for cautious, preliminary, plain-language observations —
never a diagnosis.

This is explicitly a *preliminary observation* tool, not a diagnostic
one. Every output carries a hard disclaimer and is written to avoid
diagnostic-sounding language (the prompt forbids words like "diagnosis",
"confirmed", "you have X").

Structured output (Tier 2 upgrade): the model returns strict JSON —
{finding_location, finding_description, confidence_qualifier,
differential_hint, is_xray} — parsed into a Pydantic-style dataclass,
the same `response_mime_type: application/json` pattern already used in
app/agents/ocr_agent.py, rather than one unstructured paragraph. This
lets downstream consumers (the API response, the session-findings store,
and the pipeline's Vision -> Symptom Checker -> Pharmacist chain — see
app/pipeline.py) work with discrete fields instead of parsing free text.

Only available when LLM_PROVIDER=gemini and a real GEMINI_API_KEY is
set — Grok's public API does not currently offer a vision endpoint, and
there is no safe local fallback for image analysis, so this agent
degrades to a clear "unavailable" result rather than guessing.
"""
import json
import logging
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger("medagent.vision_agent")

VISION_SYSTEM_PROMPT = """You are a Vision/Diagnostic Agent giving PRELIMINARY, NON-DIAGNOSTIC
observations on a chest X-ray image for a clinical decision support demo tool.

Return STRICT JSON only, in this exact shape:
{"is_xray": true | false,
 "finding_location": "short phrase naming the anatomical region observed (e.g. 'right lower lung zone'), or null if there is nothing notable to localize",
 "finding_description": "one or two plain-language sentences describing the visual observation, using cautious qualifiers",
 "confidence_qualifier": "a short phrase, e.g. 'low confidence', 'moderate confidence', 'appears clear' — never a numeric score",
 "differential_hint": "one short, explicitly non-diagnostic phrase (e.g. 'could be consistent with early-stage infection') for what the observation might be consistent with, or null if there is nothing notable"}

Strict rules:
- Never say "diagnosis", "confirmed", "you have", or state a disease as fact anywhere,
  including in differential_hint — phrase any hint as "could be consistent with", never
  as a conclusion.
- finding_description must use a confidence qualifier inline as well (e.g. "appears",
  "may suggest", "is not clearly visible in this image").
- If the image does not look like a chest X-ray at all, set is_xray to false and explain
  in finding_description instead of guessing at findings.
"""

DEFAULT_DISCLAIMER = (
    "This is a preliminary, non-diagnostic AI observation. It must be reviewed "
    "by a licensed radiologist or physician before any clinical decision is made."
)


@dataclass
class VisionResult:
    available: bool
    is_xray: bool = True
    finding_location: str | None = None
    finding_description: str | None = None
    confidence_qualifier: str | None = None
    differential_hint: str | None = None
    unavailable_reason: str | None = None
    disclaimer: str = DEFAULT_DISCLAIMER
    raw_json: dict = field(default_factory=dict)

    @property
    def findings(self) -> str:
        """Composed multi-line plain-text summary — kept for backward
        compatibility with consumers that want one string (the existing
        bullet-list UI rendering, and as the fallback evidence text if a
        caller doesn't want to handle the structured fields directly)."""
        if not self.available:
            return self.unavailable_reason or "Vision analysis is not available."
        if not self.is_xray:
            return self.finding_description or "The uploaded image does not appear to be a chest X-ray."

        lines = []
        if self.finding_location:
            lines.append(f"Location: {self.finding_location}")
        if self.finding_description:
            lines.append(self.finding_description)
        if self.confidence_qualifier:
            lines.append(f"Confidence: {self.confidence_qualifier}")
        if self.differential_hint:
            lines.append(f"Possible differential (non-diagnostic hint): {self.differential_hint}")
        lines.append(self.disclaimer)
        return "\n".join(lines)


class VisionAgent:
    def __init__(self):
        self.available = settings.LLM_PROVIDER == "gemini" and bool(settings.GEMINI_API_KEY)
        self._model = None
        if self.available:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(settings.VISION_MODEL_NAME)

    def analyze_xray(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> VisionResult:
        if not self.available:
            return VisionResult(
                available=False,
                unavailable_reason=(
                    "Vision analysis is not available: it requires LLM_PROVIDER=gemini "
                    "with a real GEMINI_API_KEY set (Grok does not currently expose a "
                    "free vision endpoint in this app)."
                ),
            )
        try:
            response = self._model.generate_content(
                [VISION_SYSTEM_PROMPT, {"mime_type": mime_type, "data": image_bytes}],
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            result = VisionResult(
                available=True,
                is_xray=bool(data.get("is_xray", True)),
                finding_location=data.get("finding_location"),
                finding_description=data.get("finding_description"),
                confidence_qualifier=data.get("confidence_qualifier"),
                differential_hint=data.get("differential_hint"),
                raw_json=data,
            )
            logger.info(f"Vision Agent produced a structured finding (is_xray={result.is_xray}).")
            return result
        except Exception as e:
            logger.error(f"Vision Agent call failed: {e}")
            return VisionResult(
                available=False,
                unavailable_reason=f"Vision analysis failed due to an error: {e}",
            )

    def as_finding_text(self, result: VisionResult) -> str:
        """Formats a VisionResult for storage via app.database.save_finding(),
        so a later chat turn in the same session can reference it — same
        role as OCRAgent.as_finding_text() in app/agents/ocr_agent.py."""
        return result.findings


vision_agent = VisionAgent()
