from __future__ import annotations
from typing import Literal, List, Dict, Any, Optional
import json
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState

# ---- Result schema the graph can consume ----
class CritiqueResult(BaseModel):
    action: Literal["proceed", "revise"]               # what the graph should do next
    reason: Optional[str] = Field(default="", max_length=1500)   



def planMoveCritiqueAgent(vendor: vendorState) -> vendorState:
    """
    LLM critique agent that judges ONLY the *internal logic* of the plannerAgent's rationale text.
    It does NOT use metrics, dataframes, or recompute math—just sanity-checks contradictions
    and misuse of inequalities inside the rationale itself.

    Outcome:
      - attaches `moveCritique` dict to the latest plan
      - sets `route` to "revise" (if unsound) or "proceed" (if sound)
    """
    if not vendor.container_plans:
        return vendor

    plan: ContainerPlanState = vendor.container_plans[-1]
    move = getattr(plan, "moveProposal", None)
    rationale = getattr(move, "rationale", None) if move else None

    if not rationale or not isinstance(rationale, str):
        critique = CritiqueResult(
            action="revise",
            reasons=["No rationale text found from plannerAgent to critique."],
        )
        setattr(plan, "moveCritique", critique.model_dump())
        setattr(plan, "route", critique.action)
        return vendor

    # ---- LLM: internal-consistency critique on the rationale text only ----
    system_msg = (
        "You are a strict consistency checker. Your job is to analyze ONLY the provided rationale text, "
        "and determine whether its claims are logically consistent with each other. "
        "Do NOT invent new numbers. Do NOT reference any external data. "
        "Flag contradictions (e.g., claiming 42.03% is less than 20%), broken arithmetic shown in text, "
        "or misapplied if/then rules as *unsound*. If everything is self-consistent, mark it *sound*."
    )

    output_schema = """
Return ONLY JSON:
{
  "action":  "proceed" | "revise",
  "reason": "short bullet explaining each flaw or 'no contradictions found'",
}
Rules for you:
- Treat any numeric comparisons *inside the rationale* as claims to verify against the same sentence(s).
- If a sentence asserts both A and B (e.g., 'util is 42.03%' AND 'it is < 20%'), that is a contradiction.
- If you cannot find enough information inside the rationale to prove a claim, still judge only by the text; do NOT assume.
"""

    user_msg = f"""
Rationale to critique (analyze this text ONLY):

\"\"\"{rationale}\"\"\"

Your task:
- Check internal consistency of the rationale (numbers vs. inequalities; claims vs. thresholds mentioned).
- Do not add any numbers. Do not fetch data.
- Output JSON per the schema above.
{output_schema}
"""

    client = OpenAI()
    raw = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    ).choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
        critique = CritiqueResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        critique = CritiqueResult(
            action="revise",
            reason="Critique LLM did not return valid JSON with the required fields.",
        )

    # Attach result and a routing hint
    setattr(plan, "moveCritique", critique.model_dump())
    setattr(plan, "route", critique.action)  # "proceed" or "revise"
    return vendor



def planMoveCritiqueAgent_router(vendor: vendorState) -> str:
    plan: ContainerPlanState = vendor.container_plans[-1]
    move = getattr(plan, "moveProposal", None)
    critique = getattr(move, "moveCritique", None)
    action = str(getattr(critique, "action", "") or "").lower()
    return action


