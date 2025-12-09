from __future__ import annotations
from typing import Dict, List, Optional, Any
from math import ceil, floor
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


def _cbm_in_container(plan: ContainerPlanState, container_id: int) -> float:
    """Calculate total CBM currently assigned in a container."""
    return sum((r.cbm_assigned or 0.0) for r in plan.container_plan_rows if r.container == container_id)


def _build_sku_info_map(sku_states: List[ChewySkuState]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup map for SKU MOQ and MCP info."""
    sku_map = {}
    for sku in sku_states:
        sku_map[str(sku.product_part_number)] = {
            "MOQ": int(sku.MOQ or 0),
            "MCP": int(sku.MCP or 1),
            "planned_demand": float(sku.planned_demand or 0.0),
            "runrate_at_LT_plus4w": float(sku.runrate_at_LT_plus4w or 0.0),
        }
    return sku_map


def _calculate_dos_goal(df_projections: pd.DataFrame, percentile: float = 75.0) -> float:
    """
    Calculate DOS goal as the Nth percentile of revised_DOS_end_LT_days_plus4w
    for all SKUs with non-zero planned_demand.
    
    Args:
        df_projections: DataFrame with revised projections
        percentile: The percentile to use (default 75th)
    
    Returns:
        DOS goal value
    """
    # Filter to SKUs with non-zero planned_demand
    df_filtered = df_projections[
        (df_projections["planned_demand"].notna()) & 
        (df_projections["planned_demand"] > 0)
    ].copy()
    
    if df_filtered.empty:
        return 0.0
    
    # Get valid DOS values
    dos_values = df_filtered["revised_DOS_end_LT_days_plus4w"].dropna()
    
    if dos_values.empty:
        return 0.0
    
    return float(np.percentile(dos_values, percentile))


def _get_container_sku_details(
    plan: ContainerPlanState,
    target_container: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Get SKU details for items in the target container.
    Returns dict: {ppn -> {cases_assigned, cbm_per_case, mcp, rows}}
    """
    container_skus: Dict[str, Dict[str, Any]] = {}
    
    for row in plan.container_plan_rows:
        if row.container != target_container:
            continue
        if row.cases_assigned is None or row.cases_assigned <= 0:
            continue
        
        ppn = str(row.product_part_number)
        cbm_per_case = float(row.case_pk_CBM or 0.0)
        mcp = int(row.master_case_pack or 1)
        
        if ppn not in container_skus:
            container_skus[ppn] = {
                "cases_assigned": 0,
                "cbm_per_case": cbm_per_case,
                "mcp": mcp,
                "rows": [],
            }
        
        container_skus[ppn]["cases_assigned"] += int(row.cases_assigned)
        container_skus[ppn]["rows"].append(row)
    
    return container_skus


def _calculate_cases_to_meet_dos(
    current_oh_plus_supply: float,
    target_dos: float,
    runrate: float,
    mcp: int,
    current_cases_in_container: int,
    moq: int,
) -> int:
    """
    Calculate how many cases can be trimmed from a SKU to meet the target DOS,
    while respecting MOQ and MCP.
    
    Returns: Number of cases to TRIM (positive = remove cases, 0 = no trim)
    """
    if runrate <= 0 or target_dos <= 0 or mcp <= 0:
        return 0
    
    # Target OH to achieve target DOS
    target_oh = target_dos * runrate
    
    # Current OH from original + supply in plan
    oh_reduction_needed = current_oh_plus_supply - target_oh
    
    if oh_reduction_needed <= 0:
        return 0  # Already at or below target DOS
    
    # Convert OH reduction to cases (each case = mcp eaches)
    cases_reduction = oh_reduction_needed / mcp
    
    # Round down to MCP multiple
    mcp_multiples_to_trim = floor(cases_reduction / mcp)
    cases_to_trim = mcp_multiples_to_trim * mcp
    
    # Can't trim more than what's in the container
    cases_to_trim = min(cases_to_trim, current_cases_in_container)
    
    # Ensure we don't go below MOQ
    remaining_cases = current_cases_in_container - cases_to_trim
    moq_cases = ceil(moq / mcp) if mcp > 0 else 0
    
    if remaining_cases < moq_cases and moq_cases > 0:
        # Adjust trim to leave at least MOQ
        cases_to_trim = max(0, current_cases_in_container - moq_cases)
        # Snap to MCP multiple
        cases_to_trim = (cases_to_trim // mcp) * mcp
    
    return max(0, cases_to_trim)


def _execute_trim_for_sku(
    sku_details: Dict[str, Any],
    cases_to_trim: int,
) -> float:
    """
    Execute trimming for a single SKU across its rows.
    Returns: CBM trimmed
    """
    if cases_to_trim <= 0:
        return 0.0
    
    rows = sku_details["rows"]
    cbm_per_case = sku_details["cbm_per_case"]
    mcp = sku_details["mcp"]
    
    remaining_to_trim = cases_to_trim
    total_cbm_trimmed = 0.0
    
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
        
        cbm_trimmed = trim_from_row * cbm_per_case
        total_cbm_trimmed += cbm_trimmed
        remaining_to_trim -= trim_from_row
    
    return total_cbm_trimmed


def execute_dos_based_trim(
    plan: ContainerPlanState,
    df_projections: pd.DataFrame,
    sku_info_map: Dict[str, Dict[str, Any]],
    target_container: int,
    cbm_goal: float,
    dos_goal: float,
) -> float:
    """
    Execute DOS-based trimming on target container.
    
    Iteratively trims SKUs with highest revised_DOS_end_LT_days_plus4w down to dos_goal,
    while respecting MOQ and MCP requirements.
    
    Args:
        plan: The container plan to modify
        df_projections: DataFrame with revised projections
        sku_info_map: SKU info lookup (MOQ, MCP, etc.)
        target_container: Container ID to trim from
        cbm_goal: Maximum CBM that can be trimmed (trim budget)
        dos_goal: Target DOS to trim down to
    
    Returns:
        Total CBM trimmed
    """
    # Get current SKU details in container
    container_skus = _get_container_sku_details(plan, target_container)
    
    if not container_skus:
        return 0.0
    
    # Filter projections to SKUs in container, sorted by DOS descending
    df = df_projections[
        df_projections["product_part_number"].isin(container_skus.keys())
    ].copy()
    
    if df.empty:
        return 0.0
    
    # Sort by revised_DOS_end_LT_days_plus4w descending (highest DOS first)
    df = df.sort_values("revised_DOS_end_LT_days_plus4w", ascending=False, na_position="last")
    
    total_cbm_trimmed = 0.0
    cbm_remaining = cbm_goal
    
    for _, row in df.iterrows():
        if cbm_remaining <= 1e-6:
            break
        
        ppn = str(row["product_part_number"])
        current_dos = row.get("revised_DOS_end_LT_days_plus4w")
        
        # Skip if already at or below DOS goal
        if current_dos is None or pd.isna(current_dos) or current_dos <= dos_goal:
            continue
        
        sku_details = container_skus.get(ppn)
        if not sku_details:
            continue
        
        sku_info = sku_info_map.get(ppn, {})
        mcp = int(sku_info.get("MCP", 1) or 1)
        moq = int(sku_info.get("MOQ", 0) or 0)
        runrate = float(row.get("runrate_at_LT_plus4w", 0) or 0)
        revised_oh = float(row.get("revised_projected_OH_end_LT_plus4w", 0) or 0)
        
        # Calculate how many cases to trim to meet DOS goal
        cases_to_trim = _calculate_cases_to_meet_dos(
            current_oh_plus_supply=revised_oh,
            target_dos=dos_goal,
            runrate=runrate,
            mcp=mcp,
            current_cases_in_container=sku_details["cases_assigned"],
            moq=moq,
        )
        
        if cases_to_trim <= 0:
            continue
        
        # Limit by remaining CBM budget
        cbm_per_case = sku_details["cbm_per_case"]
        max_cases_by_cbm = floor(cbm_remaining / cbm_per_case) if cbm_per_case > 0 else 0
        max_cases_by_cbm = (max_cases_by_cbm // mcp) * mcp  # Snap to MCP
        cases_to_trim = min(cases_to_trim, max_cases_by_cbm)
        
        if cases_to_trim <= 0:
            continue
        
        # Execute trim
        cbm_trimmed = _execute_trim_for_sku(sku_details, cases_to_trim)
        total_cbm_trimmed += cbm_trimmed
        cbm_remaining -= cbm_trimmed
        
        # Update tracking
        sku_details["cases_assigned"] -= cases_to_trim
    
    return total_cbm_trimmed


# ---------------------------- main entrypoint ---------------------------- #

def planMoveTrimExecutorAgent(vendor: vendorState) -> vendorState:
    """
    An agent that executes DOS-based trimming on a container.
    
    Algorithm:
    1. Calculate cbm_goal from current container CBM (trim budget)
    2. Calculate revised projections (revised_DOS_end_LT_days_plus4w) for all SKUs
    3. Set DOS_goal to 75th percentile of revised_DOS_end_LT_days_plus4w 
       (only SKUs with non-zero planned_demand)
    4. Iteratively trim SKUs with highest DOS down to DOS_goal, respecting MOQ/MCP
    5. Continue trimming until cbm_goal is met
    6. If candidates exhausted but cbm_remaining > 0, reduce DOS_goal by 10% and repeat
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
    
    sku_states: List[ChewySkuState] = getattr(vendor, "ChewySku_info", []) or []
    if not sku_states:
        return vendor
    
    # Step 1: Calculate cbm_goal from current container CBM
    cbm_goal = _cbm_in_container(plan, target_container)
    
    if cbm_goal <= 1e-6:
        return vendor
    
    # Build SKU info map (MOQ, MCP, planned_demand)
    sku_info_map = _build_sku_info_map(sku_states)
    
    # Add planned_demand to projection calculation
    df_projections = calculate_revised_projections(plan, sku_states)
    
    # Add planned_demand from sku_info_map to df_projections
    df_projections["planned_demand"] = df_projections["product_part_number"].map(
        lambda x: sku_info_map.get(str(x), {}).get("planned_demand", 0.0)
    )
    
    # Step 2 & 3: Calculate DOS goal (75th percentile)
    dos_goal = _calculate_dos_goal(df_projections, percentile=75.0)
    
    if dos_goal <= 0:
        # Fallback: use median if 75th percentile is invalid
        dos_goal = _calculate_dos_goal(df_projections, percentile=50.0)
    
    if dos_goal <= 0:
        return vendor  # No valid DOS goal, cannot proceed
    
    # Step 4, 5, 6: Iteratively trim with DOS goal reduction
    total_cbm_trimmed = 0.0
    cbm_remaining = cbm_goal
    max_iterations = 10  # Safety limit
    dos_reduction_factor = 0.10  # Reduce DOS goal by 10% each iteration
    
    print(f"[TrimAgent] Starting DOS-based trim. CBM goal: {cbm_goal:.2f}, Initial DOS goal: {dos_goal:.2f}")
    
    for iteration in range(max_iterations):
        if cbm_remaining <= 1e-6:
            break
        
        print(f"[TrimAgent] Iteration {iteration + 1}: DOS goal = {dos_goal:.2f}, CBM remaining = {cbm_remaining:.2f}")
        
        # Recalculate projections after each iteration (since assignments changed)
        if iteration > 0:
            df_projections = calculate_revised_projections(plan, sku_states)
            df_projections["planned_demand"] = df_projections["product_part_number"].map(
                lambda x: sku_info_map.get(str(x), {}).get("planned_demand", 0.0)
            )
        
        # Execute DOS-based trim
        cbm_trimmed = execute_dos_based_trim(
            plan=plan,
            df_projections=df_projections,
            sku_info_map=sku_info_map,
            target_container=target_container,
            cbm_goal=cbm_remaining,
            dos_goal=dos_goal,
        )
        
        total_cbm_trimmed += cbm_trimmed
        cbm_remaining -= cbm_trimmed
        
        print(f"[TrimAgent] Trimmed {cbm_trimmed:.2f} CBM this iteration. Total: {total_cbm_trimmed:.2f}")
        
        # If no CBM was trimmed this iteration, reduce DOS goal and retry
        if cbm_trimmed < 1e-6:
            dos_goal = dos_goal * (1 - dos_reduction_factor)
            print(f"[TrimAgent] No trim possible, reducing DOS goal by {dos_reduction_factor*100:.0f}% to {dos_goal:.2f}")
            
            if dos_goal < 1.0:  # Minimum DOS threshold
                print("[TrimAgent] DOS goal below minimum, stopping")
                break
    
    print(f"[TrimAgent] Completed. Total CBM trimmed: {total_cbm_trimmed:.2f}")
    
    # Cleanup: remove rows with zero assignments
    plan.container_plan_rows = [
        r for r in plan.container_plan_rows if int(r.cases_assigned or 0) > 0
    ]
    
    return vendor
