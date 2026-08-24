"""
3.1 Coordinator Agent

Understands the user's query, identifies intent, decides which tools are
required, and routes the query. Uses the LLM for classification, with a
deterministic heuristic fallback (keyword + known-medicine matching) so
routing still works sensibly even in mock mode or on a parsing failure.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from app.llm_client import llm_client
from app.database import list_all_medicine_names

logger = logging.getLogger("medagent.coordinator")

VALID_INTENTS = {"medicine_info", "drug_interaction", "complex", "general"}

CLASSIFY_SYSTEM_PROMPT = """You are the Coordinator Agent of a medication information system.
Classify the user's medication-related query and extract medicine names.

Return STRICT JSON only, in this exact shape:
{"intent": "medicine_info" | "drug_interaction" | "complex" | "general",
 "medicines": ["Drug1", "Drug2"],
 "reasoning": "one short sentence"}

Rules:
- "medicine_info": question about a single medicine (usage, side effects, warnings).
- "drug_interaction": user asks if two (or more) medicines can be taken together / interact.
- "complex": needs both general info AND interaction checking, or multiple medicines with broader questions.
- "general": not about a specific medicine (greeting, unrelated question, or unclear).
- Always extract medicine names exactly as the user wrote them, capitalized normally.
"""


@dataclass
class RoutingDecision:
    intent: str
    medicines: list[str] = field(default_factory=list)
    reasoning: str = ""
    required_tools: list[str] = field(default_factory=list)


class CoordinatorAgent:
    def __init__(self):
        self._known_medicines: list[str] | None = None  # lazy-loaded (DB may not be ready yet)

    @property
    def known_medicines(self) -> list[str]:
        if self._known_medicines is None:
            self._known_medicines = [m.lower() for m in list_all_medicine_names()]
        return self._known_medicines

    def route(self, user_query: str) -> RoutingDecision:
        # In mock mode (no LLM key configured) the LLM can't actually read the
        # query, so go straight to the deterministic heuristic classifier —
        # this keeps the whole pipeline demoable end-to-end with zero cost.
        if llm_client.mock_mode:
            decision = self._heuristic_classify(user_query)
        else:
            decision = self._llm_classify(user_query)
            if decision is None:
                decision = self._heuristic_classify(user_query)

        # Attach required tools based on intent (3.1 "Determining which tools are required")
        if decision.intent == "medicine_info":
            decision.required_tools = ["rag", "external"]
        elif decision.intent == "drug_interaction":
            decision.required_tools = ["sql", "interaction_tool", "external"]
        elif decision.intent == "complex":
            decision.required_tools = ["rag", "sql", "interaction_tool", "external"]
        else:
            decision.required_tools = []

        logger.info(f"Routing decision: {decision}")
        return decision

    def _llm_classify(self, user_query: str) -> RoutingDecision | None:
        try:
            raw = llm_client.complete(
                CLASSIFY_SYSTEM_PROMPT, f"User query: {user_query}", json_mode=True
            )
            data = json.loads(raw)
            intent = data.get("intent", "general")
            if intent not in VALID_INTENTS:
                intent = "general"
            medicines = data.get("medicines", []) or []
            return RoutingDecision(
                intent=intent,
                medicines=[m.strip() for m in medicines if m.strip()],
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"LLM classification failed, using heuristic fallback: {e}")
            return None

    def _heuristic_classify(self, user_query: str) -> RoutingDecision:
        """Deterministic fallback: match known medicine names + interaction keywords."""
        query_lower = user_query.lower()
        found = [m for m in self.known_medicines if re.search(rf"\b{re.escape(m)}\b", query_lower)]

        interaction_keywords = ["together", "interact", "interaction", "combine", "combination", "with"]
        has_interaction_language = any(k in query_lower for k in interaction_keywords)

        if len(found) >= 2 and has_interaction_language:
            intent = "drug_interaction"
        elif len(found) >= 2:
            intent = "complex"
        elif len(found) == 1:
            intent = "medicine_info"
        else:
            intent = "general"

        # Restore original casing from known medicine list where possible
        display_names = [name.title() for name in found]

        return RoutingDecision(
            intent=intent,
            medicines=display_names,
            reasoning="heuristic keyword/medicine-name fallback classification",
        )


coordinator_agent = CoordinatorAgent()
