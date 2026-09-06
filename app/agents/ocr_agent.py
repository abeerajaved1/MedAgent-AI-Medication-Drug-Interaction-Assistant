"""
Prescription / Label OCR Agent (new capability — the blueprint promises
OCR prescription reading; this fulfils it with near-zero new engineering
by reusing the same Gemini multimodal endpoint already wired up for the
Vision Agent's chest X-ray analysis, see app/agents/vision_agent.py).

Given a photo of a pill bottle label or a prescription slip, extracts:
  - medicine name(s) it can read
  - dosage / strength, if visible
  - frequency / instructions, if visible
  - raw OCR-style transcription of any text it can read

This is a new INPUT MODALITY (image -> structured medicine data) rather
than a new output claim — it does not diagnose or verify the
prescription; it only reads what is printed and flags low-confidence or
unreadable text rather than guessing.

Same availability constraint as the Vision Agent: requires
LLM_PROVIDER=gemini with a real GEMINI_API_KEY (Grok has no public
vision endpoint at the time of writing).
"""
import json
import logging
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger("medagent.ocr_agent")

OCR_SYSTEM_PROMPT = """You are a Prescription/Label OCR Agent. You are given a photo of either
a prescription slip or a medicine/pill bottle label. Read only what is visibly printed or
written — do not guess or invent information that isn't legible.

Return STRICT JSON only, in this exact shape:
{"medicines": [
    {"name": "as printed", "dosage": "as printed or null", "frequency": "as printed or null"}
  ],
 "raw_text": "best-effort transcription of all legible text on the image",
 "legible": true | false,
 "notes": "one short sentence, e.g. flagging illegible handwriting or a blurry image"
}

Rules:
- If handwriting is illegible, say so in "notes" and leave the relevant field null rather
  than guessing a plausible-sounding but unverified value.
- Never infer a diagnosis, indication, or medical advice from the prescription — extraction
  only.
- If the image does not appear to be a prescription or medicine label at all, set
  "legible": false, "medicines": [], and explain in "notes".
"""


@dataclass
class OCRMedicineEntry:
    name: str
    dosage: str | None = None
    frequency: str | None = None


@dataclass
class OCRResult:
    available: bool
    medicines: list[OCRMedicineEntry] = field(default_factory=list)
    raw_text: str = ""
    legible: bool = False
    notes: str = ""
    disclaimer: str = (
        "This is an automated reading of the photographed text, not a verified or "
        "clinically reviewed prescription. Always confirm medicine names and dosages "
        "with your pharmacist before taking anything."
    )


class OCRAgent:
    def __init__(self):
        self.available = settings.LLM_PROVIDER == "gemini" and bool(settings.GEMINI_API_KEY)
        self._model = None
        if self.available:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(settings.VISION_MODEL_NAME)

    def read_label(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> OCRResult:
        if not self.available:
            return OCRResult(
                available=False,
                notes=(
                    "OCR is not available: it requires LLM_PROVIDER=gemini with a real "
                    "GEMINI_API_KEY set."
                ),
            )
        try:
            response = self._model.generate_content(
                [OCR_SYSTEM_PROMPT, {"mime_type": mime_type, "data": image_bytes}],
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
            )
            data = json.loads(response.text)
            medicines = [
                OCRMedicineEntry(
                    name=m.get("name", "").strip(),
                    dosage=m.get("dosage"),
                    frequency=m.get("frequency"),
                )
                for m in data.get("medicines", []) if m.get("name", "").strip()
            ]
            result = OCRResult(
                available=True,
                medicines=medicines,
                raw_text=data.get("raw_text", ""),
                legible=bool(data.get("legible", False)),
                notes=data.get("notes", ""),
            )
            logger.info(f"OCR Agent extracted {len(medicines)} medicine entr(y/ies); legible={result.legible}")
            return result
        except Exception as e:
            logger.error(f"OCR Agent call failed: {e}")
            return OCRResult(available=False, notes=f"OCR failed due to an error: {e}")

    def as_finding_text(self, result: OCRResult) -> str:
        """Formats an OCR result for storage via app.database.save_finding(),
        so a later chat turn in the same session can reference the medicines
        that were read from a photographed label/prescription."""
        if not result.medicines:
            return f"OCR could not confidently extract medicine names. Notes: {result.notes}"
        parts = ["Medicines read from a photographed label/prescription (unverified OCR extraction):"]
        for m in result.medicines:
            desc = m.name
            if m.dosage:
                desc += f", dosage: {m.dosage}"
            if m.frequency:
                desc += f", frequency: {m.frequency}"
            parts.append(f"- {desc}")
        return "\n".join(parts)


ocr_agent = OCRAgent()
