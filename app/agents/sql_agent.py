"""
3.3 SQL Database Agent

Provides structured access to the MEDICINES, DRUG_INTERACTIONS, and
SAFETY_WARNINGS tables. The AI system calls this agent whenever a query
needs structured (rather than free-text) medication data.
"""
import logging
from dataclasses import dataclass, field

from app import database as db

logger = logging.getLogger("medagent.sql_agent")


@dataclass
class SQLLookupResult:
    medicine_info: dict | None = None
    warnings: list[dict] = field(default_factory=list)
    interaction: dict | None = None
    found: bool = False


class SQLAgent:
    def lookup_medicine(self, name: str) -> SQLLookupResult:
        row = db.find_medicine(name)
        warnings = db.find_warnings(name)
        result = SQLLookupResult(
            medicine_info=dict(row) if row else None,
            warnings=[dict(w) for w in warnings],
            found=row is not None,
        )
        logger.info(f"SQL lookup for '{name}': found={result.found}, warnings={len(warnings)}")
        return result

    def lookup_interaction(self, drug_a: str, drug_b: str) -> SQLLookupResult:
        row = db.find_interaction(drug_a, drug_b)
        result = SQLLookupResult(
            interaction=dict(row) if row else None,
            found=row is not None,
        )
        logger.info(f"SQL interaction lookup '{drug_a}' <-> '{drug_b}': found={result.found}")
        return result


sql_agent = SQLAgent()
