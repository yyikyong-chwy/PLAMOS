from __future__ import annotations
from typing import Dict, List, Optional, Any
from math import ceil
from collections import defaultdict
import pandas as pd

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


def _cbm_in_container(plan: ContainerPlanState, container_id: int) -> float:
    return sum((r.cbm_assigned or 0.0) for r in plan.container_plan_rows if r.container == container_id)


def identify_trim_candidates(
    plan: ContainerPlanState,
    df_projections: pd.DataFrame,
    target_container: int,
) -> pd.DataFrame:
    """
    Identify and rank SKUs in target container for trimming based on:
    1. High revised_DOS_end_LT_days_plus4w (85% weight) - more excess = better to trim
    2. Low PRODUCT_MARGIN_PER_UNIT (10% weight) - lower margin = better to trim
    3. High case_pk_CBM (5% weight) - larger CBM trims faster
    
    Only considers SKUs that are currently assigned in the target container.
    Returns DataFrame sorted by composite trim_score (best candidates first).
    """
    # Get SKUs currently in the target container with their assigned cases
    container_skus: Dict[str, int] = {}
    container_cbm_per_case: Dict[str, float] = {}
    container_mcp: Dict[str, int] = {}
    
    for row in plan.container_plan_rows:
        if row.container != target_container:
            continue
        if row.cases_assigned is None or row.cases_assigned <= 0:
            continue
        ppn = str(row.product_part_number)
        container_skus[ppn] = container_skus.get(ppn, 0) + int(row.cases_assigned)
        container_cbm_per_case[ppn] = float(row.case_pk_CBM or 0.0)
        container_mcp[ppn] = int(row.master_case_pack or 1)
    
    if not container_skus:
        return pd.DataFrame()
    
    # Filter projections to only SKUs in the target container
    df = df_projections[df_projections["product_part_number"].isin(container_skus.keys())].copy()
    
    if df.empty:
        return df
    
    # Add container-specific info
    df["cases_in_container"] = df["product_part_number"].map(container_skus)
    df["container_cbm_per_case"] = df["product_part_number"].map(container_cbm_per_case)
    df["container_mcp"] = df["product_part_number"].map(container_mcp)
    df["cbm_in_container"] = df["cases_in_container"] * df["container_cbm_per_case"]
    
    # Filter out invalid entries
    df = df[(df["cases_in_container"] > 0) & (df["container_cbm_per_case"] > 0)].copy()
    
    if df.empty:
        return df
    
    # Normalize factors for scoring
    
    # DOS plus 4w: higher is better for trimming (more excess)
    dos_col = df["revised_DOS_end_LT_days_plus4w"].fillna(0)
    dos_max = dos_col.max() if dos_col.max() > 0 else 1
    df["dos_score"] = dos_col / dos_max  # Higher DOS = higher score = better to trim
    
    # Margin: lower is better for trimming (keep high-margin items)
    margin_col = df["PRODUCT_MARGIN_PER_UNIT"].fillna(0)
    margin_max = margin_col.max() if margin_col.max() > 0 else 1
    df["margin_score"] = 1 - (margin_col / margin_max)  # Lower margin = higher score = better to trim
    
    # CBM: higher is better (trims faster)
    cbm_col = df["container_cbm_per_case"].fillna(0)
    cbm_max = cbm_col.max() if cbm_col.max() > 0 else 1
    df["cbm_score"] = cbm_col / cbm_max  # Higher CBM = higher score
    
    # Composite trim score (weighted combination)
    # Weights: DOS_plus4w (85%), Margin (10%), CBM (5%)
    df["trim_score"] = (
        0.85 * df["dos_score"] +
        0.10 * df["margin_score"] +
        0.05 * df["cbm_score"]
    )
    
    # Sort by trim_score descending (best trim candidates first)
    df = df.sort_values("trim_score", ascending=False).reset_index(drop=True)
    
    return df


def execute_trim(
    plan: ContainerPlanState,
    df_candidates: pd.DataFrame,
    target_container: int,
    cbm_goal: float,
) -> float:
    """
    Execute trimming on target container using cbm_goal as the trim budget.
    Removes cases from SKUs based on trim_score ranking.
    Respects MCP multiples when trimming.
    
    Args:
        cbm_goal: The trim budget (maximum CBM that can be trimmed from container)
    
    Returns the actual CBM trimmed.
    """
    current_cbm = _cbm_in_container(plan, target_container)
    # cbm_goal is the trim budget (how much we're allowed to trim)
    cbm_to_trim = min(cbm_goal, current_cbm)
    
    if cbm_to_trim <= 1e-9:
        return 0.0
    
    total_trimmed_cbm = 0.0
    cbm_remaining_to_trim = cbm_to_trim
    
    # Build lookup: SKU -> list of rows in target container
    rows_by_sku: Dict[str, List[ContainerPlanRow]] = defaultdict(list)
    for r in plan.container_plan_rows:
        if r.container == target_container and (r.cases_assigned or 0) > 0:
            rows_by_sku[str(r.product_part_number)].append(r)
    
    candidates = df_candidates.to_dict("records")
    
    # Pass 1: Trim by MCP chunks, proportionally by score
    for c in candidates:
        if cbm_remaining_to_trim <= 1e-6:
            break
        
        ppn = c["product_part_number"]
        mcp = int(c.get("container_mcp", 1) or 1)
        cbm_per_case = float(c.get("container_cbm_per_case", 0.0) or 0.0)
        cases_available = int(c.get("cases_in_container", 0) or 0)
        
        if mcp <= 0 or cbm_per_case <= 0 or cases_available <= 0:
            continue
        
        rows = rows_by_sku.get(ppn, [])
        if not rows:
            continue
        
        # Calculate how many cases to trim (in MCP multiples)
        mcp_chunk_cbm = mcp * cbm_per_case
        mcp_chunks_needed = ceil(cbm_remaining_to_trim / mcp_chunk_cbm)
        max_mcp_chunks_available = cases_available // mcp
        
        mcp_chunks_to_trim = min(mcp_chunks_needed, max_mcp_chunks_available)
        cases_to_trim = mcp_chunks_to_trim * mcp
        
        if cases_to_trim <= 0:
            continue
        
        # Distribute trim across rows for this SKU
        remaining_to_trim = cases_to_trim
        for row in rows:
            if remaining_to_trim <= 0:
                break
            
            row_cases = int(row.cases_assigned or 0)
            trim_from_row = min(remaining_to_trim, row_cases)
            
            # Snap to MCP multiple for this row
            trim_from_row = (trim_from_row // mcp) * mcp
            if trim_from_row <= 0:
                continue
            
            row.cases_assigned = row_cases - trim_from_row
            row.cbm_assigned = float(row.cases_assigned) * cbm_per_case
            
            trimmed_cbm = trim_from_row * cbm_per_case
            total_trimmed_cbm += trimmed_cbm
            cbm_remaining_to_trim = max(0.0, cbm_remaining_to_trim - trimmed_cbm)
            remaining_to_trim -= trim_from_row
    
    return total_trimmed_cbm


# ---------------------------- main entrypoint ---------------------------- #

def planMoveTrimExecutorAgent(vendor: vendorState) -> vendorState:
    """
    An agent that executes the trim move proposal in vendor.container_plans[-1].moveProposal.
    It will perform the following steps:
        1. Based on the current plan, reanalyze the projected onhand and DOS for each SKU
        2. Identify SKUs to trim from the target container based on:
            - High revised_DOS_end_LT_days_plus4w (85% weight) - more excess = trim first
            - Low PRODUCT_MARGIN_PER_UNIT (10% weight) - keep high-margin items
            - High case_pk_CBM (5% weight) - larger CBM trims faster
        3. Execute trimming using current container CBM as trim budget, respecting MCP multiples
        
    Note: cbm_goal is computed from current container CBM (not from LLM) to avoid hallucinations.
    """
    if not vendor.container_plans:
        return vendor

    plan: ContainerPlanState = vendor.container_plans[-1]
    move: Dict = getattr(plan, "moveProposal", None) or {}
    
    if not move:
        return vendor
    
    mv_type = str(_safe_get(move, "action", "") or "").lower()
    if mv_type != "trim":
        return vendor
    
    # Get target container from move proposal
    target_container = _safe_get(move, "trim.container", None) or _safe_get(move, "trim.from_container", None)
    if target_container is None:
        return vendor
    target_container = int(target_container)
    
    # This is useless since LLM tends to hallucinate the cbm_goal
    # cbm_goal = _safe_get(move, "trim.cbm_goal", None) or _safe_get(move, "trim.target_cbm", None)
    cbm_goal = _cbm_in_container(plan, target_container)
    
    sku_states: List[ChewySkuState] = getattr(vendor, "ChewySku_info", []) or []
    if not sku_states:
        return vendor
    
    # Step 1: Calculate revised projections based on current plan
    df_projections = calculate_revised_projections(plan, sku_states)
    
    # Step 2: Identify trim candidates in target container
    df_candidates = identify_trim_candidates(plan, df_projections, target_container)
    
    if df_candidates.empty:
        return vendor
    
    # Step 3: Execute the trim
    execute_trim(plan, df_candidates, target_container, cbm_goal)
    
    # Cleanup: remove rows with zero assignments
    plan.container_plan_rows = [
        r for r in plan.container_plan_rows if int(r.cases_assigned or 0) > 0
    ]
    
    return vendor

