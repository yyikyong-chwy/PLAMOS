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
ALMOST_FULL_MIN = 0.70
VERY_LOW_UTIL = 0.20
TEMP = 0

ActionType = Literal["reduce", "consolidate", "pad"]

def apply_prompt_rules(vendor: vendorState):

    startegy = vendor.container_plans[-1].strategy
    rule_prompt = ""
    if startegy == PlanStrategy.CONSOLIDATE_REDUCE:
        rule_prompt = f"""
        RULES (exactly ONE move: reduce | consolidate | pad)

        Status types:
        - FULL: the container is full.
        - NOT_QUITE_FULL: the container is not quite full, but close to full.
        - LOW_UTIL: the container is low utilized.
        - PARTIAL_UTIL: the container is partially utilized, but not low utilized.

        Move types:
        - reduce: propose a CBM goal to remove.
        - consolidate: move all or portion (cbm_move) from a partial container into another partial one.
        - pad: add CBM goal to a specific container.

        RULES:
        1. Never propose anything on a container that has status FULL.

        2. if ALL containers have status FULL, propose do_nothing.

        3. If there are more than one container that is PARTIAL_UTIL or LOW_UTIL, propose to consolidate them. 
        Propose to move as much as possible from the least utilized container to the next least utilized container. 
        If MDT is one of the containers, then propose to consolidate with other destinations.             

        4. if ALL but one containers have status FULL, and that single container is LOW_UTIL, propose REDUCE the former container to remove it.
        
        5. if ALL but one containers have status FULL, and that single container is NOT_QUITE_FULL, propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.
                        
        6. if none of rule 1 to 5 applies, propose do_nothing.
        """
    elif startegy == PlanStrategy.CONSOLIDATE_ONLY:
        rule_prompt += f"""
        RULES (exactly ONE move:  consolidate | pad | do_nothing)

        Status types:
        - FULL: the container is full.
        - NOT_QUITE_FULL: the container is not quite full, but close to full.
        - LOW_UTIL: the container is low utilized.
        - PARTIAL_UTIL: the container is partially utilized, but not low utilized.

        Move types:
        - consolidate: move all or portion (cbm_move) from a partial container into another partial one.
        - pad: add CBM goal to a specific container.

        Decision:
        1. If there are more than one container that is PARTIAL_UTIL or LOW_UTIL, 
            a) starting with the least utilized container, propose move as much as possible from the least utilized container to the next least utilized container 
            b) if the combined CBM is less than CBM_Max, propose to move all CBM from the least utilized container to the next least utilized container
            c) if MDT is one of the containers, then propose to consolidate with other destinations.

        2. If all containers are FULL and only one container is NOT_QUITE_FULL:
            a) Propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.

        3. if none of rule 1 to 2 applies, propose do_nothing.
        """
    elif startegy == PlanStrategy.CONSOLIDATE_AND_PAD:
        rule_prompt += f"""
        RULES (exactly ONE move:  consolidate | pad | do_nothing)

        Status types:
        - FULL: the container is full.
        - NOT_QUITE_FULL: the container is not quite full, but close to full.
        - LOW_UTIL: the container is low utilized.
        - PARTIAL_UTIL: the container is partially utilized, but not low utilized.

        Move types:
        - consolidate: move all or portion (cbm_move) from a partial container into another partial one.
        - pad: add CBM goal to a specific container.

        Decision:
        1. If there are more than one container that is PARTIAL_UTIL or LOW_UTIL, 
            a) starting with the least utilized container, propose move as much as possible from the least utilized container to the next least utilized container 
            b) if the combined CBM is less than CBM_Max, propose to move all CBM from the least utilized container to the next least utilized container

        2. If all containers are FULL and **only** one container is not status FULL:
            a) Propose PAD to reach ~{int(FULL_THRESHOLD*100)}%.

        3. if none of rule 1 to 2 applies, propose do_nothing.
        """
    elif startegy == PlanStrategy.PAD_ONLY:
        rule_prompt += f"""
        RULES (exactly ONE move:  pad | do_nothing)

        Move types:
        - pad: add CBM goal to a specific container.

        Decision:
        1. If any container is not statusFULL, propose PAD one of them randomlyto reach ~{int(FULL_THRESHOLD*100)}%.

        2. If no container is not status FULL, propose do_nothing.
        """


    #finally add the output and explanation requirement
    rule_prompt += f"""
        ## OUTPUT & EXPLANATION REQUIREMENT

        - Provide **only** the JSON as specified below.
        - **If you propose any consolidation(s)**, your **rationale must explicitly state which rule is applied and why.**
        for **each** consolidation, using this pattern (containers, DESTs, status):
        - Example: "The two least utilized containers (container 2 in MDT1 and container 5 in TLA1) can be consolidated as their status is not FULL."
        - whatever move you propose, you must first explain how you come to this conclusion and which rule 1, 2a, 2b is applied.You need to justify how that rule is applicable in this scenario

        Return JSON ONLY:
        {{
        "action": "reduce" | "consolidate" | "pad" | "do_nothing",
        "rationale": "string",
        "reduce": {{"cbm_goal": 0.0}} | null,
        "consolidate": {{"from_dest": "", "from_container": 0, "to_dest": "", "to_container": 0, }} | null,
        "pad": {{"dest": "", "container": 0, "cbm_goal": 0.0}} | null,
        "do_nothing": null
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
    cols = ["DEST", "container", "cbm_used", "capacity_cbm", "unused_cbm", "status"]
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

def _get_latest_critique(plan: ContainerPlanState) -> Optional[dict]:
    """Fetch any critique attached to the latest move proposal.
    """
    move = getattr(plan, "moveProposal", None)
    critique = getattr(move, "moveCritique", None) if move else None
    return critique

def build_planner_prompt(vendor: vendorState, metrics: ContainerPlanMetrics) -> str:
    cbm_max = float(getattr(vendor, "CBM_Max", 66.0) or 66.0)
    cont_df = _containers_df(metrics)
    container_utilization_status_info_msg = metrics.container_utilization_status_info

    rules = apply_prompt_rules(vendor)



    #taking out the containers df for now
    #CONTAINERS: {_safe_json(cont_df.to_dict(orient='records'))}
    context = f"""
    VENDOR: {vendor.vendor_Code}, CBM_Max={cbm_max}
    CONTAINER UTILIZATION STATUS: {container_utilization_status_info_msg}
    """
    return f"You are a planning analyst. Suggest ONE move based on the rules.\n\n{rules}\n\n{context}"


def plannerAgent(vendor: vendorState) -> OneMoveProposal:
    if not vendor.container_plans:
        return OneMoveProposal(action="reduce", rationale="No plan available", reduce=Reduce(cbm_goal=0.0))

    plan: ContainerPlanState = vendor.container_plans[-1]
    #df = plan.to_df()
    metrics = plan.metrics
    
    prompt = build_planner_prompt(vendor, metrics)
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
        print(vendor.vendor_Code)
        print(data['rationale'])
        print(pd.DataFrame(vendor.container_plans[-1].metrics.total_cbm_used_by_container_dest))
        print(pd.DataFrame(data))
        
        moveProposal = OneMoveProposal.model_validate(data)        
        vendor.container_plans[-1].moveProposal = moveProposal
        return vendor
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        vendor.container_plans[-1].moveProposal = OneMoveProposal(action="reduce", rationale=f"Parse error: {e}", reduce=Reduce(cbm_goal=0.0))
        return vendor
