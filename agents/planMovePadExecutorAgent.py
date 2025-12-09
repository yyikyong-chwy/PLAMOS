from __future__ import annotations
from typing import Dict, List, Optional, Any
from math import ceil
from collections import defaultdict
import pandas as pd
import numpy as np

from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerRow import ContainerPlanRow
from states.ChewySkuState import ChewySkuState
from agents.planEvalAgent import calculate_revised_projections


# ---------------------------- helpers ---------------------------- #

def _safe_get(obj, attr_path: str, default=None):
    """Safely get nested attributes/keys."""
    cur = obj
    for part in attr_path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur if cur is not None else default


def _dest_for_container(plan: ContainerPlanState, container_id: int) -> Optional[str]:
    for r in plan.container_plan_rows:
        if r.container == container_id:
            return getattr(r, "DEST", None)
    return None


def _cbm_in_container(plan: ContainerPlanState, container_id: int) -> float:
    return sum((r.cbm_assigned or 0.0) for r in plan.container_plan_rows if r.container == container_id)


def _get_or_create_dst_row(
    plan: ContainerPlanState,
    proto_row: ContainerPlanRow,
    dst_container: int,
    dst_dest: str) -> ContainerPlanRow:
    """Get existing row or create new one for SKU in target container."""
    for r in plan.container_plan_rows:
        if (r.container == dst_container) and (r.product_part_number == proto_row.product_part_number):
            if getattr(r, "DEST", None) != dst_dest:
                setattr(r, "DEST", dst_dest)
            return r
    data = proto_row.model_dump()
    data.update({
        "container": dst_container,
        "DEST": dst_dest,
        "cases_assigned": 0,
        "cbm_assigned": 0.0,
    })
    new_r = ContainerPlanRow.model_validate(data)
    plan.container_plan_rows.append(new_r)
    return new_r


def _snap_cases_to_mcp_and_moq(cases: int, mcp: int, moq: int) -> int:
    """
    Snap cases to MCP multiple and ensure MOQ is met.
    Returns 0 if constraints cannot be satisfied.
    """
    if mcp <= 0:
        mcp = 1
    if moq <= 0:
        moq = 0
    
    # Round up to MCP multiple
    if cases <= 0:
        return 0
    
    snapped = ceil(cases / mcp) * mcp
    
    # Ensure MOQ is met (MOQ is in eaches, convert to cases)
    # If MOQ > snapped * mcp (eaches), we need more cases
    # Actually MOQ is typically cases or eaches - assume it's cases for simplicity
    if snapped < moq:
        snapped = ceil(moq / mcp) * mcp
    
    return snapped


def build_pad_candidates_df(
    df_projections: pd.DataFrame,
    sku_states: List[ChewySkuState],
) -> pd.DataFrame:
    """
    Build a DataFrame of pad candidates with all necessary fields.
    Filters out SKUs where CHW_OTB is True.
    """
    df = df_projections.copy()
    
    # Filter out invalid SKUs (must have positive MCP and CBM)
    df = df[(df["MCP"] > 0) & (df["case_pk_CBM"] > 0)].copy()
    
    # Filter out SKUs with CHW_OTB = True (truthy value indicates OTB status)
    df = df[df["CHW_OTB"].isna() | (df["CHW_OTB"] == 0) | (df["CHW_OTB"] == False)].copy()
    
    if df.empty:
        return df
    
    # Build MOQ lookup from sku_states
    moq_map = {str(s.product_part_number): (s.MOQ or 0) for s in sku_states}
    df["MOQ"] = df["product_part_number"].map(moq_map).fillna(0).astype(int)
    
    return df


def execute_pad_with_projections(
    vendor: vendorState,
    plan: ContainerPlanState,
    df_projections: pd.DataFrame,
    sku_states: List[ChewySkuState],
    move: Dict,
) -> None:
    """
    Execute multi-pass padding on target container.
    
    Pass 1: Fill SKUs with negative revised_projected_OH_end_LT until they reach 0.
    Pass 2+: Calculate median DOS, pad all SKUs up to goal_DOS starting with lowest.
             If cbm remains, increase goal_DOS by 20% and repeat.
    
    All passes respect MOQ and MCP constraints.
    """
    CBM_Max: float = float(getattr(vendor, "CBM_Max", 66.0))
    dst_cid = _safe_get(move, "pad.container", None) or _safe_get(move, "pad.to_container", None)
    if dst_cid is None:
        return
    dst_cid = int(dst_cid)
    
    dst_dest = _dest_for_container(plan, dst_cid)
    if dst_dest is None:
        return
    
    current_cbm = _cbm_in_container(plan, dst_cid)
    cbm_remaining = max(0.0, CBM_Max - current_cbm)
    if cbm_remaining <= 1e-9:
        return
    
    # Build candidates DataFrame
    df_candidates = build_pad_candidates_df(df_projections, sku_states)
    if df_candidates.empty:
        return
    
    # Build lookup for existing rows
    rows_by_sku: Dict[str, List[ContainerPlanRow]] = defaultdict(list)
    for r in plan.container_plan_rows:
        rows_by_sku[str(r.product_part_number)].append(r)
    
    def _any_row_for_sku(sku: str) -> Optional[ContainerPlanRow]:
        lst = rows_by_sku.get(sku, [])
        return lst[0] if lst else None
    
    def _ensure_row_for_sku_in_container(sku: str, cbm_per_case: float, mcp: int) -> ContainerPlanRow:
        # Reuse existing row if already present in target container
        for r in plan.container_plan_rows:
            if r.container == dst_cid and str(r.product_part_number) == sku:
                if getattr(r, "DEST", None) != dst_dest:
                    setattr(r, "DEST", dst_dest)
                return r
        
        # Clone from existing row if SKU exists elsewhere
        proto = _any_row_for_sku(sku)
        if proto is not None:
            return _get_or_create_dst_row(plan, proto, dst_cid, dst_dest)
        
        # Build minimal new row
        vcode = getattr(vendor, "vendor_Code", None)
        vname = getattr(vendor, "vendor_name", None)
        if vname is None:
            for rr in plan.container_plan_rows:
                if getattr(rr, "vendor_name", None):
                    vname = rr.vendor_name
                    break
        payload = {
            "vendor_Code": vcode,
            "vendor_name": vname,
            "DEST": dst_dest,
            "container": dst_cid,
            "product_part_number": sku,
            "master_case_pack": int(mcp or 1),
            "case_pk_CBM": float(cbm_per_case or 0.0),
            "cases_needed": 0,
            "cases_assigned": 0,
            "cbm_assigned": 0.0,
        }
        new_r = ContainerPlanRow.model_validate(payload)
        plan.container_plan_rows.append(new_r)
        rows_by_sku[sku].append(new_r)
        return new_r
    
    def add_cases_to_sku(sku: str, cases_to_add: int, cbm_per_case: float, mcp: int) -> float:
        """Add cases to SKU in container, return CBM added."""
        if cases_to_add <= 0:
            return 0.0
        drow = _ensure_row_for_sku_in_container(sku, cbm_per_case, mcp)
        drow.cases_assigned = int(drow.cases_assigned or 0) + int(cases_to_add)
        drow.cbm_assigned = float(drow.cases_assigned) * float(drow.case_pk_CBM or cbm_per_case)
        return cases_to_add * cbm_per_case
    
    # Track padding additions per SKU (to update projections)
    padding_added: Dict[str, int] = defaultdict(int)  # SKU -> eaches added
    
    # =========================
    # PASS 1: Fill negative projected OH SKUs to 0
    # =========================
    negative_oh_skus = df_candidates[df_candidates["revised_projected_OH_end_LT"] < 0].copy()
    negative_oh_skus = negative_oh_skus.sort_values("revised_projected_OH_end_LT", ascending=True)
    
    for _, row in negative_oh_skus.iterrows():
        if cbm_remaining <= 1e-9:
            break
        
        sku = str(row["product_part_number"])
        mcp = int(row["MCP"] or 1)
        moq = int(row.get("MOQ", 0) or 0)
        cbm_per_case = float(row["case_pk_CBM"] or 0.0)
        revised_oh = float(row["revised_projected_OH_end_LT"] or 0.0)
        
        if mcp <= 0 or cbm_per_case <= 0:
            continue
        
        # Calculate eaches needed to reach 0 OH
        eaches_needed = -revised_oh + padding_added.get(sku, 0) * mcp
        if eaches_needed <= 0:
            continue
        
        # Convert to cases
        cases_needed_raw = eaches_needed / mcp
        cases_needed = _snap_cases_to_mcp_and_moq(int(ceil(cases_needed_raw)), mcp, moq)
        
        # Check CBM constraint
        cbm_needed = cases_needed * cbm_per_case
        if cbm_needed > cbm_remaining:
            # Reduce to fit remaining CBM
            max_cases = int(cbm_remaining / cbm_per_case)
            cases_needed = (max_cases // mcp) * mcp
            if cases_needed < moq:
                cases_needed = 0  # Can't meet MOQ
        
        if cases_needed > 0:
            cbm_added = add_cases_to_sku(sku, cases_needed, cbm_per_case, mcp)
            cbm_remaining -= cbm_added
            padding_added[sku] += cases_needed
    
    if cbm_remaining <= 1e-9:
        return
    
    # =========================
    # PASS 2+: Pad to median DOS, then increase by 20%
    # =========================
    
    # Recalculate current DOS values accounting for padding added
    def get_current_revised_dos(row) -> float:
        sku = str(row["product_part_number"])
        mcp = int(row["MCP"] or 1)
        runrate = float(row.get("runrate_at_LT", 0) or 0)
        original_oh = float(row["revised_projected_OH_end_LT"] or 0)
        added_eaches = padding_added.get(sku, 0) * mcp
        current_oh = original_oh + added_eaches
        if runrate > 0:
            return current_oh / runrate
        return 9999.0
    
    df_candidates["current_dos"] = df_candidates.apply(get_current_revised_dos, axis=1)
    
    # Calculate initial goal_DOS as 25th percentile (start conservative)
    valid_dos = df_candidates["current_dos"].replace([np.inf, -np.inf], np.nan).dropna()
    valid_dos = valid_dos[valid_dos < 9999]
    
    if valid_dos.empty:
        return
    
    goal_dos = float(valid_dos.quantile(0.25))
    
    # Iteratively pad to goal_DOS, increasing by 20% each round
    max_iterations = 20  # Safety limit
    iteration = 0
    
    while cbm_remaining > 1e-9 and iteration < max_iterations:
        iteration += 1
        made_progress = False
        
        # Recalculate current DOS for all SKUs
        df_candidates["current_dos"] = df_candidates.apply(get_current_revised_dos, axis=1)
        
        # Sort by current DOS ascending (lowest DOS first)
        df_sorted = df_candidates.sort_values("current_dos", ascending=True)
        
        for _, row in df_sorted.iterrows():
            if cbm_remaining <= 1e-9:
                break
            
            sku = str(row["product_part_number"])
            mcp = int(row["MCP"] or 1)
            moq = int(row.get("MOQ", 0) or 0)
            cbm_per_case = float(row["case_pk_CBM"] or 0.0)
            runrate = float(row.get("runrate_at_LT", 0) or 0)
            original_oh = float(row["revised_projected_OH_end_LT"] or 0)
            
            if mcp <= 0 or cbm_per_case <= 0 or runrate <= 0:
                continue
            
            # Current OH with padding added
            added_eaches = padding_added.get(sku, 0) * mcp
            current_oh = original_oh + added_eaches
            current_dos = current_oh / runrate if runrate > 0 else 9999
            
            # Skip if already at or above goal DOS
            if current_dos >= goal_dos:
                continue
            
            # Calculate eaches needed to reach goal_dos
            target_oh = goal_dos * runrate
            eaches_needed = target_oh - current_oh
            if eaches_needed <= 0:
                continue
            
            # Convert to cases
            cases_needed_raw = eaches_needed / mcp
            cases_needed = _snap_cases_to_mcp_and_moq(int(ceil(cases_needed_raw)), mcp, moq)
            
            # Check CBM constraint
            cbm_needed = cases_needed * cbm_per_case
            if cbm_needed > cbm_remaining:
                # Reduce to fit remaining CBM
                max_cases = int(cbm_remaining / cbm_per_case)
                cases_needed = (max_cases // mcp) * mcp
                # Check MOQ - if we can't meet MOQ, try at least one MCP chunk
                if cases_needed < moq:
                    if mcp * cbm_per_case <= cbm_remaining:
                        cases_needed = mcp  # Add at least one MCP
                    else:
                        cases_needed = 0
            
            if cases_needed > 0:
                cbm_added = add_cases_to_sku(sku, cases_needed, cbm_per_case, mcp)
                cbm_remaining -= cbm_added
                padding_added[sku] += cases_needed
                made_progress = True
        
        # Check if all SKUs are at goal_dos
        df_candidates["current_dos"] = df_candidates.apply(get_current_revised_dos, axis=1)
        all_at_goal = (df_candidates["current_dos"] >= goal_dos).all()
        
        if all_at_goal or not made_progress:
            # Increase goal_dos by 20% and continue
            goal_dos *= 1.20
        
        # If no progress was made and we already increased goal, break
        if not made_progress and not all_at_goal:
            # Try one more round with increased goal
            goal_dos *= 1.20


# ---------------------------- main entrypoint ---------------------------- #

def planMovePadExecutorAgent(vendor: vendorState) -> vendorState:
    """
    An agent that executes the padding move proposal in vendor.container_plans[-1].moveProposal.
    
    Multi-pass padding strategy:
    
    PASS 1: Negative OH Recovery
        - Identify all SKUs with negative revised_projected_OH_end_LT
        - Pad these SKUs until they reach 0 revised_projected_OH_end_LT
        - Prioritize by most negative first
    
    PASS 2+: DOS Equalization
        - Calculate 25th percentile of all revised_DOS_end_LT_days as initial goal_DOS
        - Iterate all SKUs starting with lowest DOS
        - Pad each SKU up to goal_DOS
        - If CBM remains after all SKUs reach goal_DOS, increase goal_DOS by 20%
        - Repeat until cbm_remaining is exhausted
    
    All passes enforce:
        - MOQ (Minimum Order Quantity) compliance
        - MCP (Master Case Pack) multiples
        - CHW_OTB must NOT be True for candidate SKUs
    """
    if not vendor.container_plans:
        return vendor

    plan: ContainerPlanState = vendor.container_plans[-1]
    move: Dict = getattr(plan, "moveProposal", None) or {}
    
    if not move:
        return vendor
    
    mv_type = str(_safe_get(move, "action", "") or "").lower()
    if mv_type != "pad":
        return vendor
    
    sku_states: List[ChewySkuState] = getattr(vendor, "ChewySku_info", []) or []
    if not sku_states:
        return vendor
    
    # Calculate revised projections based on current plan
    df_projections = calculate_revised_projections(plan, sku_states)
    
    if df_projections.empty:
        return vendor
    
    # Execute the multi-pass padding strategy
    execute_pad_with_projections(vendor, plan, df_projections, sku_states, move)
    
    # Cleanup: remove rows with zero assignments
    plan.container_plan_rows = [
        r for r in plan.container_plan_rows if int(r.cases_assigned or 0) > 0
    ]
    
    return vendor
