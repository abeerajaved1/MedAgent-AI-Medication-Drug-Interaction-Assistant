"""
Phase 2 (free approach — no fine-tuning/GPU required):

The original blueprint's SO3 called for QLoRA fine-tuning a 7-8B model to
teach "pharmacist-style reasoning" (age -> weight -> allergies ->
comorbidities -> differential -> recommendation -> warnings) instead of
just drug memorization. That needs real GPU-hours and a training
pipeline that doesn't belong in a free web app.

The free, zero-training substitute used here: a structured
chain-of-thought PROMPT TEMPLATE that forces the same reasoning steps
onto a strong instruct model (Gemini/Grok) at inference time. This is a
legitimate, well-documented technique (prompted CoT reasoning) — not a
fine-tuned model, and the write-up should describe it honestly as that:
a prompting-based reasoning scaffold, not a trained reasoning adapter.

This module is imported by the Pharmacist Agent and injected into its
system prompt whenever patient-profile context (age/allergies/
conditions) is available, so the reasoning chain has real inputs to
reason over.
"""

REASONING_CHAIN_TEMPLATE = """
Follow this structured reasoning process before writing your final answer
(you may think through it silently, but your final answer should reflect
having reasoned through each relevant step):

1. AGE — Does the patient's age (if known) change what's appropriate or
   safe here? (e.g. pediatric dosing limits, elderly sensitivity)
2. ALLERGIES — Does the patient have any stated allergy that conflicts
   with the medicine(s) in question?
3. CURRENT CONDITIONS — Do any stated conditions (e.g. kidney disease,
   pregnancy, liver disease) change the safety picture?
4. CURRENT MEDICATIONS — Could this interact with anything already on
   the patient's medication list?
5. EVIDENCE CHECK — Is every claim you're about to make backed by the
   evidence you were given? Do not add anything the evidence doesn't
   support.
6. RECOMMENDATION — State what the evidence shows plainly and calmly.
7. WARNINGS — Explicitly surface any warning uncovered in steps 1-4,
   even if the user didn't ask about it directly.

If patient profile context (age/allergies/conditions/current
medications) was NOT provided, skip steps 1-4 silently and say so only
if it's relevant to the question (e.g. "age-specific dosing would need
your age or a doctor's input").
"""


def build_patient_context_block(profile: dict | None) -> str:
    """Formats stored patient-profile fields into a short context block
    for the reasoning chain to reason over. Returns "" if no profile."""
    if not profile:
        return ""
    parts = []
    if profile.get("age") is not None:
        parts.append(f"Age: {profile['age']}")
    if profile.get("allergies"):
        parts.append(f"Known allergies: {', '.join(profile['allergies'])}")
    if profile.get("conditions"):
        parts.append(f"Existing conditions: {', '.join(profile['conditions'])}")
    if profile.get("current_medications"):
        parts.append(f"Current medications: {', '.join(profile['current_medications'])}")
    if not parts:
        return ""
    return "Patient context (self-reported, not clinically verified):\n" + "\n".join(f"- {p}" for p in parts)
