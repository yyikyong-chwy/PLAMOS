from __future__ import annotations
from typing import Dict, List, Optional, Any
from math import floor
from collections import defaultdict
import pandas as pd

from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerRow import ContainerPlanRow
from states.ChewySkuState import ChewySkuState


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


def calculate_revised_projections(
    plan: ContainerPlanState,
    sku_states: List[ChewySkuState],
) -> pd.DataFrame:
    """
    Calculate revised projected inventory based on current plan assignments.
    
    Returns a DataFrame with:
    - product_part_number
    - additional_supply_eaches (from current plan)
    - revised_projected_OH_end_LT
    - revised_projected_OH_end_LT_plus4w
    - revised_DOS_end_LT_days
    - revised_DOS_end_LT_days_plus4w
    - F90_DAILY_AVG
    - PRODUCT_MARGIN_PER_UNIT
    - case_pk_CBM
    - MCP
    """
    # Aggregate supply by SKU from current plan (cases_assigned * master_case_pack)
    supply_by_sku: Dict[str, int] = {}
    for row in plan.container_plan_rows:
        if row.cases_assigned is None or row.cases_assigned == 0:
            continue
        ppn = row.product_part_number
        eaches = row.cases_assigned * row.master_case_pack
        supply_by_sku[ppn] = supply_by_sku.get(ppn, 0) + eaches
    
    records = []
    for sku in sku_states:
        ppn = sku.product_part_number
        additional_eaches = supply_by_sku.get(ppn, 0)
        
        # Original projections
        original_oh_end_lt = sku.projected_OH_end_LT or 0.0
        original_oh_end_lt_plus4w = sku.projected_OH_end_LT_plus4w or 0.0
        
        # Revised projections = original + additional supply
        revised_oh_end_lt = original_oh_end_lt + additional_eaches
        revised_oh_end_lt_plus4w = original_oh_end_lt_plus4w + additional_eaches
        
        # Calculate revised DOS
        daily_avg = sku.F90_DAILY_AVG or sku.T90_DAILY_AVG or 0.0
        revised_dos_end_lt = revised_oh_end_lt / daily_avg if daily_avg > 0 else None
        revised_dos_end_lt_plus4w = revised_oh_end_lt_plus4w / daily_avg if daily_avg > 0 else None
        
        records.append({
            "product_part_number": ppn,
            "additional_supply_eaches": additional_eaches,
            "original_projected_OH_end_LT": original_oh_end_lt,
            "revised_projected_OH_end_LT": revised_oh_end_lt,
            "original_projected_OH_end_LT_plus4w": original_oh_end_lt_plus4w,
            "revised_projected_OH_end_LT_plus4w": revised_oh_end_lt_plus4w,
            "revised_DOS_end_LT_days": revised_dos_end_lt,
            "revised_DOS_end_LT_days_plus4w": revised_dos_end_lt_plus4w,
            "F90_DAILY_AVG": daily_avg,
            "PRODUCT_MARGIN_PER_UNIT": sku.PRODUCT_MARGIN_PER_UNIT or 0.0,
            "case_pk_CBM": sku.case_pk_CBM or 0.0,
            "MCP": sku.MCP or 0,
        })
    
    return pd.DataFrame(records)


def identify_promising_pad_skus(
    df_projections: pd.DataFrame,
    vendor: vendorState,
) -> pd.DataFrame:
    """
    Identify and rank promising SKUs for padding based on:
    1. SKUs with no excess demand (revised_projected_OH_end_LT <= 0 or low DOS)
    2. Least projected DOS at end of lead time (prioritize low DOS)
    3. Largest margin per unit
    4. Larger CBM per case (helps fill container faster)
    
    Returns DataFrame sorted by composite score (best candidates first).
    """
    df = df_projections.copy()
    
    # Filter out invalid SKUs (must have positive MCP and CBM)
    df = df[(df["MCP"] > 0) & (df["case_pk_CBM"] > 0)].copy()
    
    if df.empty:
        return df
    
    # Calculate excess indicator: negative or low projected OH indicates need
    # We want SKUs that are NOT in excess (revised_projected_OH_end_LT <= some threshold)
    # Lower DOS = more urgent need = better candidate
    df["has_no_excess"] = df["revised_projected_OH_end_LT"] <= 0
    
    # Normalize factors for scoring (handle edge cases)
    # DOS: lower is better -> invert for scoring
    dos_col = df["revised_DOS_end_LT_days"].fillna(9999)
    dos_max = dos_col.max() if dos_col.max() > 0 else 1
    df["dos_score"] = 1 - (dos_col / dos_max)  # Higher score for lower DOS
    
    # Margin: higher is better
    margin_max = df["PRODUCT_MARGIN_PER_UNIT"].max() if df["PRODUCT_MARGIN_PER_UNIT"].max() > 0 else 1
    df["margin_score"] = df["PRODUCT_MARGIN_PER_UNIT"] / margin_max
    
    # CBM: higher is better (fills container faster)
    cbm_max = df["case_pk_CBM"].max() if df["case_pk_CBM"].max() > 0 else 1
    df["cbm_score"] = df["case_pk_CBM"] / cbm_max
    
    # Composite score (weighted combination)
    # Weights: DOS (40%), Margin (35%), CBM (25%)
    df["pad_score"] = (
        0.60 * df["dos_score"] +
        0.30 * df["margin_score"] +
        0.10 * df["cbm_score"]
    )
    
    # Boost score for SKUs with no excess (they definitely need stock)
    df.loc[df["has_no_excess"], "pad_score"] += 0.5
    
    # Sort by score descending (best candidates first)
    df = df.sort_values("pad_score", ascending=False).reset_index(drop=True)
    
    return df


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


def execute_pad_with_projections(
    vendor: vendorState,
    plan: ContainerPlanState,
    df_candidates: pd.DataFrame,
    move: Dict,
) -> None:
    """
    Execute padding on target container using ranked candidate SKUs.
    Distributes CBM across top candidates respecting MCP multiples.
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
    remaining_cbm = max(0.0, CBM_Max - current_cbm)
    if remaining_cbm <= 1e-9:
        return
    
    # Determine how many SKUs to pad with based on container utilization
    # Goal: spread risk by using more SKUs when container is emptier
    utilization_pct = (current_cbm / CBM_Max) * 100 if CBM_Max > 0 else 0
    
    if utilization_pct >= 70:
        # Container is >= 70% full -> pad with 1 SKU (small gap to fill)
        target_sku_count = 1
    elif utilization_pct >= 30:
        # Container is 30-70% full -> pad with 2 SKUs (medium gap)
        target_sku_count = 2
    else:
        # Container is < 30% full -> pad with 3 SKUs (large gap, spread risk)
        target_sku_count = 3
    
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
    
    # Use target SKU count based on utilization (bounded by available candidates)
    TOP_N = min(target_sku_count, len(df_candidates))
    if TOP_N == 0:
        return
    
    top_candidates = df_candidates.head(TOP_N).to_dict("records")
    
    # Weights from pad_score
    weights = [max(1e-9, c.get("pad_score", 0.01)) for c in top_candidates]
    total_w = sum(weights)
    cbm_left = remaining_cbm
    
    # Pass 1: Proportional allocation by weight
    for i, c in enumerate(top_candidates):
        if cbm_left <= 0.1:
            break
        sku = c["product_part_number"]
        mcp = int(c.get("MCP", 1) or 1)
        cbm_case = float(c.get("case_pk_CBM", 0.0) or 0.0)
        if mcp <= 0 or cbm_case <= 0:
            continue
        
        w = weights[i] / total_w
        target_cbm = cbm_left * w
        
        max_cases_by_target = int(target_cbm // cbm_case)
        max_cases_by_cap = int(cbm_left // cbm_case)
        max_cases = max(0, min(max_cases_by_target, max_cases_by_cap))
        
        # Snap to MCP multiple
        cases_to_add = (max_cases // mcp) * mcp
        if cases_to_add <= 0:
            continue
        
        drow = _ensure_row_for_sku_in_container(sku, cbm_case, mcp)
        drow.cases_assigned = int(drow.cases_assigned or 0) + int(cases_to_add)
        drow.cbm_assigned = float(drow.cases_assigned) * float(drow.case_pk_CBM or cbm_case)
        
        cbm_added = cases_to_add * cbm_case
        cbm_left = max(0.0, cbm_left - cbm_added)
    
    # Pass 2: Round-robin fill leftover (one MCP chunk at a time)
    made_progress = True
    while cbm_left >= 1e-6 and made_progress:
        made_progress = False
        for c in top_candidates:
            if cbm_left < 1e-6:
                break
            sku = c["product_part_number"]
            mcp = int(c.get("MCP", 1) or 1)
            cbm_case = float(c.get("case_pk_CBM", 0.0) or 0.0)
            if mcp <= 0 or cbm_case <= 0:
                continue
            
            mcp_chunk_cbm = mcp * cbm_case
            if mcp_chunk_cbm <= cbm_left + 1e-9:
                drow = _ensure_row_for_sku_in_container(sku, cbm_case, mcp)
                drow.cases_assigned = int(drow.cases_assigned or 0) + mcp
                drow.cbm_assigned = float(drow.cases_assigned) * float(drow.case_pk_CBM or cbm_case)
                cbm_left = max(0.0, cbm_left - mcp_chunk_cbm)
                made_progress = True


# ---------------------------- main entrypoint ---------------------------- #

def planMovePadExecutorAgent(vendor: vendorState) -> vendorState:
    """
    An agent that executes the padding move proposal in vendor.container_plans[-1].moveProposal.
    It will perform the following steps:
        1. based on the current plan, it will reanalyze the projected onhand at end of lead time for each SKU
        2. based on the current plan, it will reanalyze the projected onhand at end of lead time plus 4wks for each SKU
        3. it will identify promising skus to pad up to the target container
            the considerations to determine the promising skus are:
            - skus with no excess demand (<= 0)
            - skus with the least projected DOS at end of lead time
            - skus with the largest margin per unit
            - skus with the larger cbm per case (to help padding up quicker)
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
    
    # Step 1 & 2: Reanalyze projected OH at end of LT and LT+4wks based on current plan
    df_projections = calculate_revised_projections(plan, sku_states)
    
    # Step 3: Identify promising SKUs to pad
    df_candidates = identify_promising_pad_skus(df_projections, vendor)
    
    if df_candidates.empty:
        return vendor
    
    # Execute the padding with the ranked candidates
    execute_pad_with_projections(vendor, plan, df_candidates, move)
    
    # Cleanup: remove rows with zero assignments
    plan.container_plan_rows = [
        r for r in plan.container_plan_rows if int(r.cases_assigned or 0) > 0
    ]
    
    return vendor
