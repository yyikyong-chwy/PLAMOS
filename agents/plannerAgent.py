from __future__ import annotations
from typing import Any, Dict, Optional, Literal
import json, re
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerPlanMetrics import ContainerPlanMetrics
from states.containerPlanState import PlanStrategy
from states.plannerMoveProposal import OneMoveProposal, Reduce, Consolidate, Pad

FULL_THRESHOLD = 0.95
CLOSE_TO_FULL_MIN = 0.70
VERY_LOW_UTIL = 0.20
TEMP = 0.2

ActionType = Literal["reduce", "consolidate", "pad"]

def apply_prompt_rules(vendor: vendorState):
    cbm_max = float(getattr(vendor, "CBM_Max", 66.0) or 66.0)
    startegy = vendor.container_plans[-1].strategy
    rule_prompt = ""
    if startegy == PlanStrategy.CONSOLIDATE_REDUCE:
        rule_prompt = f"""
        RULES (exactly ONE move: reduce | consolidate | pad)

        Move types:
        - reduce: propose a CBM goal to remove.
        - consolidate: move all or portion (cbm_move) from a partial container into another partial one.
        - pad: add CBM goal to a specific container.

        Decision:
        1. If there are more than one container that is less than {int(FULL_THRESHOLD*100)}% utilization, 
            a) starting with the least utilized container, propose move as much as possible from the least utilized container to the next least utilized container 
            b) if the combined CBM is less than CBM_Max, propose to move all CBM from the least utilized container to the next least utilized container

        2. If all containers are full (≥{int(FULL_THRESHOLD*100)}%) and only one container is partial:
            a) If the partial one ≥{int(CLOSE_TO_FULL_MIN*100)}%, propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.
            b) If the partial one <{int(VERY_LOW_UTIL*100)}%, propose REDUCE to remove it.
            c) Else, propose REDUCE with cbm_goal=0 (leave as is).

        3. If >1 container partial (<{int(FULL_THRESHOLD*100)}%):
            a) If two least utilized containers combined CBM ≤ CBM_Max={cbm_max}, CONSOLIDATE both fully (move everything from the lower-utilized into the other).
            b) If combined > CBM_Max, CONSOLIDATE to fill one container (~{int(FULL_THRESHOLD*100)}%), choosing target DEST with this order:
                - Default TNY1 first; but if the least-occupied container is also TNY1, then choose TLA1 **if such a partially filled container exists**, otherwise MDT1. Move only the CBM needed (cbm_move) to reach ~{int(FULL_THRESHOLD*100)}% on the chosen target.
        """
    elif startegy == PlanStrategy.CONSOLIDATE_ONLY:
        rule_prompt += f"""
        RULES (exactly ONE move:  consolidate | pad | do_nothing)

        Move types:
        - consolidate: move all or portion (cbm_move) from a partial container into another partial one.
        - pad: add CBM goal to a specific container.

        Decision:
        1. If there are more than one container that is less than {int(FULL_THRESHOLD*100)}% utilization, 
            a) starting with the least utilized container, propose move as much as possible from the least utilized container to the next least utilized container 
            b) if the combined CBM is less than CBM_Max, propose to move all CBM from the least utilized container to the next least utilized container

        2. If all containers are full (≥{int(FULL_THRESHOLD*100)}%) and only one container is partial:
            a) If the partial one ≥{int(CLOSE_TO_FULL_MIN*100)}%, propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.

        3. If >1 container partial (<{int(FULL_THRESHOLD*100)}%):
            a) If two least utilized containers combined CBM ≤ CBM_Max={cbm_max}, CONSOLIDATE both fully (move everything from the lower-utilized into the other).
            b) If combined > CBM_Max, CONSOLIDATE to fill one container (~{int(FULL_THRESHOLD*100)}%), choosing target DEST with this order:
                - Default TNY1 first; but if the least-occupied container is also TNY1, then choose TLA1 **if such a partially filled container exists**, otherwise MDT1. Move only the CBM needed (cbm_move) to reach ~{int(FULL_THRESHOLD*100)}% on the chosen target.

        4. Propose do_nothing.
        """
    elif startegy == PlanStrategy.CONSOLIDATE_AND_PAD:
        rule_prompt += f"""
        RULES (exactly ONE move:  consolidate | pad | do_nothing)

        Move types:
        - consolidate: move all or portion (cbm_move) from a partial container into another partial one.
        - pad: add CBM goal to a specific container.

        Decision:
        1. If there are more than one container that is less than {int(FULL_THRESHOLD*100)}% utilization, 
            a) starting with the least utilized container, propose move as much as possible from the least utilized container to the next least utilized container 
            b) if the combined CBM is less than CBM_Max, propose to move all CBM from the least utilized container to the next least utilized container

        2. If all containers are full (≥{int(FULL_THRESHOLD*100)}%) and **only** one container is partial:
            a) If the partial one ≥{int(CLOSE_TO_FULL_MIN*100)}%, propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.

        3. If >1 container partial (<{int(FULL_THRESHOLD*100)}%):
            a) If two least utilized containers combined CBM ≤ CBM_Max={cbm_max}, CONSOLIDATE both fully (move everything from the lower-utilized into the other).
            b) If combined > CBM_Max, CONSOLIDATE to fill one container (~{int(FULL_THRESHOLD*100)}%), choosing target DEST with this order:
                - Default TNY1 first; but if the least-occupied container is also TNY1, then choose TLA1 **if such a partially filled container exists**, otherwise MDT1. Move only the CBM needed (cbm_move) to reach ~{int(FULL_THRESHOLD*100)}% on the chosen target.

        4. If no container is partial, propose DO_NOTHING.
        """
    elif startegy == PlanStrategy.PAD_ONLY:
        rule_prompt += f"""
        RULES (exactly ONE move:  pad | do_nothing)

        Move types:
        - pad: add CBM goal to a specific container.

        Decision:
        1. If any container is partial ≥{int(CLOSE_TO_FULL_MIN*100)}%, propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.

        2. If no container is partial, propose DO_NOTHING.
        """


    #finally add the output and explanation requirement
    rule_prompt += f"""
        ## OUTPUT & EXPLANATION REQUIREMENT

        - Provide **only** the JSON as specified below.
        - **If you propose any consolidation(s)**, your **rationale must explicitly show the arithmetic** that proves feasibility against `CBM_Max`
        for **each** consolidation, using this pattern (containers, DESTs, numbers, and inequality):
        - Example: "The two least utilized containers (container 2 in MDT1 and container 5 in TLA1) can be consolidated as their combined CBM is less than CBM_Max (54.99 + 28.34 = 83.33 <= 64.0)."
        (Use the actual container IDs, DESTs, and CBM values from context.)
        - Example: "The two least utilized containers (container 3 in TLA1 with 43.89 CBM and container 4 in TNY1 with 31.59 CBM) can be consolidated. However
        as their combined CBM is more than CBM_Max (43.89 + 31.59 = 75.48 > 66.0) I propose to move 31.11 CBM from container 3 in TLA1 to container 4 in TNY1 to reach ~95% utilization.
        - If you propose to reduce, you must first explain how you come to this conclusion and how rule 1, 1a, 1b is applied. In addition, you must provide the cbm_goal and also how you calculated it. 

        Return JSON ONLY:
        {{
        "action": "reduce" | "consolidate" | "pad",
        "rationale": "string",
        "reduce": {{"cbm_goal": 0.0}} | null,
        "consolidate": {{"from_dest": "", "from_container": 0, "to_dest": "", "to_container": 0, "cbm_move": 0.0}} | null,
        "pad": {{"dest": "", "container": 0, "cbm_goal": 0.0}} | null
        }}
    """
    return rule_prompt

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), indent=2)

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if fence:
            return json.loads(fence.group(1))
        brace = re.search(r"(\{.*\})", text, flags=re.S)
        if brace:
            return json.loads(brace.group(1))
        raise ValueError("No JSON found in LLM response.")

def _containers_df(metrics: ContainerPlanMetrics) -> pd.DataFrame:
    rows = getattr(metrics, "total_cbm_used_by_container_dest", None) or []
    cols = ["DEST", "container", "cbm_used", "capacity_cbm", "unused_cbm"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def _pick_target_dest(from_dest: str, least_dest: str, partial_dests: list[str]) -> str:
    """Pick consolidation target per preference:
    - Default to TNY1 first.
    - If the least-occupied container's DEST is TNY1, then prefer TLA1 if it exists among partial_dests; otherwise MDT1.
    """
    preferred = ["TNY1", "TLA1", "MDT1"]
    if least_dest == "TNY1":
        preferred = ["TLA1", "MDT1"]
    for d in preferred:
        if d in partial_dests:
            return d
    # fallback to least_dest if no preferred dest available
    return least_dest if from_dest == "TNY1" else "TNY1"


def build_planner_prompt(vendor: vendorState, metrics: ContainerPlanMetrics, plan_df: pd.DataFrame) -> str:
    cbm_max = float(getattr(vendor, "CBM_Max", 66.0) or 66.0)
    dest_cbm = plan_df.groupby("DEST")["cbm_assigned"].sum().to_dict() if not plan_df.empty else {}
    cont_df = _containers_df(metrics)

    rules = apply_prompt_rules(vendor)

    context = f"""
    VENDOR: {vendor.vendor_Code}, CBM_Max={cbm_max}
    DEST→CBM used: {_safe_json(dest_cbm)}
    CONTAINERS: {_safe_json(cont_df.to_dict(orient='records'))}
    """
    return f"You are a planning analyst. Suggest ONE move based on the rules.\n\n{rules}\n\n{context}"


def plannerAgent(vendor: vendorState) -> OneMoveProposal:
    if not vendor.container_plans:
        return OneMoveProposal(action="reduce", rationale="No plan available", reduce=Reduce(cbm_goal=0.0))

    plan: ContainerPlanState = vendor.container_plans[-1]
    df = plan.to_df()
    metrics = plan.metrics

    if df.empty:
        return OneMoveProposal(action="reduce", rationale="Empty plan", reduce=Reduce(cbm_goal=0.0))

    prompt = build_planner_prompt(vendor, metrics, df)
    try:
        client = OpenAI()
        raw = (client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMP,
        ).choices[0].message.content or "").strip()
    except Exception as e:
        return OneMoveProposal(action="reduce", rationale=f"LLM error: {e}", reduce=Reduce(cbm_goal=0.0))

    try:
        data = _extract_json(raw)

        #debug
        print(data['rationale'])
        print(pd.DataFrame(vendor.container_plans[0].metrics.total_cbm_used_by_container_dest))
        print(pd.DataFrame(data))


        has_r, has_c, has_p = bool(data.get("reduce")), bool(data.get("consolidate")), bool(data.get("pad"))
        count = sum([has_r, has_c, has_p])
        if count != 1:
            if has_c: data.update({"action": "consolidate", "reduce": None, "pad": None})
            elif has_p: data.update({"action": "pad", "reduce": None, "consolidate": None})
            elif has_r: data.update({"action": "reduce", "consolidate": None, "pad": None})
            else: data = {"action": "reduce", "rationale": "No valid move", "reduce": {"cbm_goal": 0.0}, "consolidate": None, "pad": None}
        
        moveProposal = OneMoveProposal.model_validate(data)        
        vendor.container_plans[-1].moveProposal = moveProposal
        return vendor
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        vendor.container_plans[-1].moveProposal = OneMoveProposal(action="reduce", rationale=f"Parse error: {e}", reduce=Reduce(cbm_goal=0.0))
        return vendor
