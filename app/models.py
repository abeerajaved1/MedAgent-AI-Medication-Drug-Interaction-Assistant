from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class SourceRef(BaseModel):
    source: str
    snippet: str


class ChatResponse(BaseModel):
    query_type: str
    medicines: list[str] = []
    interaction_status: Optional[str] = None   # none | minor | moderate | major | not_found
    explanation: str
    safety_note: str
    verification_status: str                   # approved | warning | insufficient_evidence
    sources: list[SourceRef] = []
    disclaimer: str = (
        "MedAgent provides general medication information and is not a "
        "substitute for professional medical advice. Always consult a "
        "licensed doctor or pharmacist before making decisions about your "
        "medications."
    )
