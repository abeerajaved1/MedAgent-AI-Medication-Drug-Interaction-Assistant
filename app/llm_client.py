"""
Unified LLM client.

Supports two free-tier-friendly providers:
  - Gemini (Google AI Studio) via google-generativeai
  - Grok (xAI) via the OpenAI-compatible client

If no API key is configured for the selected provider, the client falls
back to a deterministic MOCK mode so the whole agent pipeline can still be
demoed / unit-tested end-to-end without spending any money or needing keys.
"""
import json
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger("medagent.llm")


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self._gemini_model = None
        self._grok_client = None
        self.mock_mode = False

        if self.provider == "gemini":
            if not settings.GEMINI_API_KEY:
                logger.warning("No GEMINI_API_KEY set — running LLM in MOCK mode.")
                self.mock_mode = True
            else:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)

        elif self.provider == "grok":
            if not settings.GROK_API_KEY:
                logger.warning("No GROK_API_KEY set — running LLM in MOCK mode.")
                self.mock_mode = True
            else:
                from openai import OpenAI
                self._grok_client = OpenAI(
                    api_key=settings.GROK_API_KEY,
                    base_url=settings.GROK_BASE_URL,
                )
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def complete(self, system_prompt: str, user_prompt: str,
                 json_mode: bool = False, temperature: float = 0.2) -> str:
        """Returns raw text response from the configured LLM."""
        if self.mock_mode:
            return self._mock_complete(system_prompt, user_prompt, json_mode)

        try:
            if self.provider == "gemini":
                return self._gemini_complete(system_prompt, user_prompt, json_mode, temperature)
            elif self.provider == "grok":
                return self._grok_complete(system_prompt, user_prompt, json_mode, temperature)
        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {e}. Falling back to mock.")
            return self._mock_complete(system_prompt, user_prompt, json_mode)

    def _gemini_complete(self, system_prompt, user_prompt, json_mode, temperature) -> str:
        generation_config = {"temperature": temperature}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._gemini_model.generate_content(
            full_prompt, generation_config=generation_config
        )
        return response.text

    def _grok_complete(self, system_prompt, user_prompt, json_mode, temperature) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._grok_client.chat.completions.create(
            model=settings.GROK_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Mock mode: lets the whole pipeline run deterministically with no
    # API key, no cost, and no network access. Used for local dev/testing
    # and as a safety-net fallback if the live API call fails.
    # ------------------------------------------------------------------
    def _mock_complete(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        if json_mode:
            if "classify" in system_prompt.lower() or "intent" in system_prompt.lower():
                return json.dumps({
                    "intent": "medicine_info",
                    "medicines": [],
                    "reasoning": "mock-mode heuristic classification",
                })
            if "verify" in system_prompt.lower() or "safety" in system_prompt.lower():
                return json.dumps({
                    "status": "approved",
                    "safety_note": "Always confirm with a licensed pharmacist or doctor.",
                    "unsupported_claims_found": False,
                })
            return json.dumps({})
        return (
            "[MOCK RESPONSE — no LLM API key configured] "
            "Based on the retrieved evidence, here is a general summary. "
            "Please configure GEMINI_API_KEY or GROK_API_KEY for real answers."
        )


llm_client = LLMClient()
