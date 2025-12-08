
from __future__ import annotations
from typing import Dict, List, Tuple, Any, TypedDict, Union
from collections import defaultdict
import pandas as pd
import numpy as np

# --- your state classes (import from your project) ---
# ContainerPlanRow: fields include vendor_Code, vendor_name, DEST, container, product_part_number,
#                   master_case_pack, case_pk_CBM, cases_needed, cases_assigned, cbm_assigned
from states.ContainerRow import ContainerPlanRow  # :contentReference[oaicite:5]{index=5}
from states.vendorState import vendorState                 # :contentReference[oaicite:7]{index=7}
from states.containerPlanState import ContainerPlanState, ContainerPlanMetrics
from states.planStrategy import PlanStrategy, _STRATEGY_ORDER
from states.ChewySkuState import ChewySkuState

# ---------------------------- shared projection helper ---------------------------- #

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
            "CHW_OTB": sku.CHW_OTB,
        })
    
    return pd.DataFrame(records)


FULL_THRESHOLD = 0.95
ALMOST_FULL_MIN = 0.70
VERY_LOW_UTIL = 0.20
PLAN_EVAL_MAX_LOOPS = 10

#simple aid functions
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, None) else 0.0

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _sku_cases_map(vendor: vendorState) -> Dict[str, Dict[str, float]]:
    """
    Build a mapping for each SKU -> {MCP, planned_cases, planned_demand_eaches}
    All demand fields are assumed to be *eaches* in ChewySkuState; convert to cases via MCP.
    Missing MCP or zero MCP -> skip those SKUs from alignment scoring.
    """
    out: Dict[str, Dict[str, float]] = {}
    for s in vendor.ChewySku_info:
        mcp = (s.MCP or 0)
        if mcp <= 0:
            continue
        def to_cases(x):
            return float(x) / float(mcp) if x is not None else None

        out[str(s.product_part_number)] = {
            "MCP": float(mcp),
            "planned_cases": to_cases(s.planned_demand),
            "planned_demand_eaches": float(s.planned_demand) if s.planned_demand is not None else 0.0,
        }
    return out

def _container_util_metrics(df: pd.DataFrame, cbm_max: float, low_thresh: float = 0.90) -> Tuple[int, float, float, float, int]:
    """
    Returns: (containers, total_used, total_capacity, weighted_util, low_util_count)
    """
    if df.empty:
        return 0, 0.0, 0.0, 0.0, 0

    # Sum cbm_assigned per container
    by_cont = df.groupby("container", dropna=True)["cbm_assigned"].sum().rename("cbm_used").reset_index()
    by_cont["cbm_used"] = by_cont["cbm_used"].fillna(0.0)
    by_cont["capacity"] = float(cbm_max)
    by_cont["util"] = by_cont["cbm_used"] / by_cont["capacity"]

    containers = len(by_cont)
    total_used = float(by_cont["cbm_used"].sum())
    total_capacity = float(containers * cbm_max)
    weighted_util = _safe_div(total_used, total_capacity)

    low_util_count = int((by_cont["util"] < low_thresh).sum())
    return containers, total_used, total_capacity, weighted_util, low_util_count

def _weighted_ape(final_cases: pd.Series, target_cases: pd.Series) -> float:
    """
    A weighted MAPE-like error in [0,1], using target as weight denominator.
    If target <= 0, that SKU is dropped from the calculation.
    """
    df = pd.DataFrame({"final": final_cases, "target": target_cases}).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["target"] > 0]
    if df.empty:
        return 0.0
    abs_pct = (df["final"] - df["target"]).abs() / df["target"]
    weights = df["target"] / df["target"].sum()
    return float((abs_pct * weights).sum())


def _weighted_underfill_penalty(final_cases: pd.Series, target_cases: pd.Series, down_weight: float = 3.0) -> float:
    """
    Weighted penalty for underfilling relative to target.
    - No penalty if final >= target
    - Penalty = (target - final) / target * down_weight when final < target
    Returns value in [0,1]; higher = worse alignment (more underfill)
    """
    df = pd.DataFrame({"final": final_cases, "target": target_cases}).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["target"] > 0]
    if df.empty:
        return 0.0

    underfill = (df["target"] - df["final"]).clip(lower=0)
    # Only penalize underfilled SKUs; multiply by down_weight
    abs_pct = (underfill / df["target"]) * down_weight

    weights = df["target"] / df["target"].sum()
    penalty = float((abs_pct * weights).sum())

    return float(np.clip(penalty, 0, 1))

def _container_status(u: float) -> str:
        # Order matters; ties go to the "more full" bucket unless specified.
        if u >= FULL_THRESHOLD:
            return "FULL"
        if u >= ALMOST_FULL_MIN:
            return "NOT_QUITE_FULL"
        if u <= VERY_LOW_UTIL:
            return "LOW_UTIL"
        return "PARTIAL_UTIL"

#this is to fight hallucination in LLM. it continually fail to understand the status of the containers and the rules.
def _generate_container_utilization_status_info(df: pd.DataFrame) -> str:
    """
    Build a message about container fullness and embed which containers are not full.
    Expects columns: ['DEST', 'container', 'status', 'util'].
      - 'status': string like 'FULL', 'CLOSE_TO_FULL', 'PARTIAL_UTIL', 'LOW_UTIL'
      - 'util': fraction in [0,1] (e.g., 0.836859)
    """
    if df is None or df.empty:
        return "no containers present"

    needed = {"DEST", "container", "status", "util"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    # Normalize status and find non-full rows
    tmp = df.copy()
    tmp["STATUS_NORM"] = tmp["status"].astype(str).str.upper()
    non_full = tmp[tmp["STATUS_NORM"] != "FULL"].copy()

    count = len(non_full)
    if count == 0:
        return "all containers are full"

    # Build labels like "TNY1-6 (ALMOST_FULL, 83.69% utilized)"
    def _label(row) -> str:
        util_pct = f"{float(row['util'])*100:.2f}%"
        return f"{row['DEST']}-{row['container']} ({row['STATUS_NORM']}, {util_pct} utilized)"

    labels = [_label(r) for _, r in non_full.iterrows()]
    label_str = ", ".join(labels)

    # Count categories (only add these if present)
    vc = non_full["STATUS_NORM"].value_counts()
    almost_full_n = int(vc.get("NOT_QUITE_FULL", 0))
    partial_n = int(vc.get("PARTIAL_UTIL", 0))
    low_util_n = int(vc.get("LOW_UTIL", 0))

    counts_bits = []
    if almost_full_n > 0:
        counts_bits.append(f"{almost_full_n} NOT_QUITE_FULL")
    if partial_n > 0:
        counts_bits.append(f"{partial_n} PARTIAL_UTIL")
    if low_util_n > 0:
        counts_bits.append(f"{low_util_n} LOW_UTIL")

    counts_suffix = f" (including {', '.join(counts_bits)})" if counts_bits else ""

    if count == 1:
        return f"there is only one container that is not full: {label_str}{counts_suffix}"
    else:
        return f"there are more than one container that is not full: {label_str}{counts_suffix}"

def planEvalAgent(vendor: vendorState) -> vendorState:

    if not vendor.container_plans:
        return vendor

    #weights to evaluate the plan, maybe arbitrary so lets change as it goes
    weights = {"util": 0.50, "planned": 0.50}    
    low_util_threshold = 0.90

    latest_plan = vendor.container_plans[-1]
    df = latest_plan.to_df()        

    # Normalize dtypes
    if not df.empty:
        for c in ("cbm_assigned", "case_pk_CBM"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        for c in ("cases_assigned", "master_case_pack"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # --- Container utilization block ---
    containers, total_used, total_capacity, weighted_util, low_util_count = _container_util_metrics(
        df, cbm_max=float(vendor.CBM_Max or 66.0), low_thresh=low_util_threshold
    )
    avg_util = 0.0
    if containers > 0:
        # average of per-container utilization (unweighted)
        per_cont_util = (df.groupby("container")["cbm_assigned"].sum() / float(vendor.CBM_Max or 66.0))
        avg_util = float(per_cont_util.mean())

    # Utilization score: reward >= 0.90; linear 0.75→0.90→1.00 ramp; penalize low-util containers
    util_core = _clamp01((weighted_util - 0.75) / (1.0 - 0.75))   # 0 at 75%, 1 at 100%
    low_frac = _safe_div(low_util_count, containers) if containers > 0 else 0.0
    util_score = util_core * (1.0 - 0.5 * low_frac)                # up to -50% if all are < threshold
    util_score = _clamp01(util_score)

    # --- SKU alignment block ---
    # Final assigned cases by SKU (sum across DEST & containers)
    final_cases_by_sku = df.groupby("product_part_number")["cases_assigned"].sum().astype(float) if not df.empty else pd.Series(dtype=float)

    # Targets from ChewySkuState (converted to cases)
    sku_map = _sku_cases_map(vendor)  # {sku: {"MCP":..., "planned_cases":..., "base_cases":..., "excess_cases":...}}

    # Build targets table correctly (dict-of-dicts -> rows = SKUs)
    if sku_map:
        target_df = pd.DataFrame.from_dict(sku_map, orient="index")
        target_df.index = target_df.index.astype(str)
    else:
        # Empty structure with expected columns avoids KeyErrors later
        target_df = pd.DataFrame(columns=["MCP", "planned_cases", "planned_demand_eaches"])
        target_df.index.name = "product_part_number"

    # Final assigned cases by SKU (sum across DEST & containers)
    final_cases_by_sku = (
    df.groupby("product_part_number")["cases_assigned"].sum().astype(float)
    if not df.empty else pd.Series(dtype=float)
    )
    # Normalize index dtype to string for guaranteed alignment
    final_cases_by_sku.index = final_cases_by_sku.index.astype(str)

    # Align both sides on the union of SKU keys so nothing is silently dropped
    combined_index = final_cases_by_sku.index.union(target_df.index)

    final_cases = final_cases_by_sku.reindex(combined_index).fillna(0.0)

    # Make sure target columns exist even if sku_map was empty
    for col in ("planned_cases", "planned_demand_eaches", "MCP"):
        if col not in target_df.columns:
            target_df[col] = np.nan
    target_df = target_df.reindex(combined_index)

    # APEs (weighted)
    ape_vs_planned = _weighted_ape(final_cases, target_df["planned_cases"])

    # Convert APE (error) to score
    planned_score = 1.0 - _clamp01(ape_vs_planned)

    # --- Overall score ---
    overall = (
        100.0
        * (weights.get("util", 0.50)   * util_score
         + weights.get("planned", 0.50)* planned_score)
    )

    
    #    Produces a list of {DEST, container, cbm_used}
    if not df.empty:
        cbm_cap = float(vendor.CBM_Max or 66.0)
        used_by_cdest_df = (
            df.groupby(["DEST", "container"], dropna=True)["cbm_assigned"]
                .sum()
                .reset_index()
                .rename(columns={"cbm_assigned": "cbm_used"})
        )
        used_by_cdest_df["capacity_cbm"] = cbm_cap
        used_by_cdest_df["util"] = used_by_cdest_df["cbm_used"] / used_by_cdest_df["capacity_cbm"]
        # Unused = max(capacity - used, 0). If a container overflows, unused=0 (overflow can be inferred via cbm_used > capacity).
        used_by_cdest_df["unused_cbm"] = (used_by_cdest_df["capacity_cbm"] - used_by_cdest_df["cbm_used"]).clip(lower=0.0)

        used_by_cdest_df["status"] = used_by_cdest_df["util"].apply(_container_status)

        total_cbm_used_by_container_dest = used_by_cdest_df.to_dict(orient="records")

        container_utilization_status_info = _generate_container_utilization_status_info(used_by_cdest_df)
    else:
        total_cbm_used_by_container_dest = [] #empty list if no df
        container_utilization_status_info = "no containers present"

    print("-------Container Utilization Status Info-----------")
    print(container_utilization_status_info)
    print("--------------------------------")

    # Placeholder for excess tracking (no longer computed as baseDemand/excess_demand removed)
    total_excess_in_cbm_by_container: Dict[Any, float] = {}

    # --- DemandMet tracking by SKU ---
    demand_met_by_sku_list: List[Dict[str, Union[str, float]]] = []
    
    # Get all SKUs from both original demand (target_df) and assigned (final_cases)
    all_skus = final_cases.index.union(target_df.index)
    
    for sku in all_skus:
        sku_str = str(sku)
        
        # Get MCP for this SKU
        mcp = 1.0
        if sku in target_df.index and "MCP" in target_df.columns:
            mcp_val = target_df.loc[sku, "MCP"]
            mcp = float(mcp_val) if pd.notna(mcp_val) and mcp_val > 0 else 1.0
        
        # Original demand in eaches (at virtual factory level, summed across all locations)
        original_demand = 0.0
        if sku in target_df.index and "planned_demand_eaches" in target_df.columns:
            planned_val = target_df.loc[sku, "planned_demand_eaches"]
            original_demand = float(planned_val) if pd.notna(planned_val) else 0.0
        
        # Assigned demand in eaches (from container plan, summed across all DEST/containers)
        assigned_demand = 0.0
        if sku in final_cases.index:
            cases_assigned_total = float(final_cases.loc[sku])
            assigned_demand = cases_assigned_total * mcp
        
        # Calculate delta
        delta = assigned_demand - original_demand
        
        demand_met_by_sku_list.append({
            "product_part_number": sku_str,
            "original_demand": original_demand,
            "assigned_demand": assigned_demand,
            "delta": delta,
        })

    # Write metrics back to the plan
    latest_plan_metrics = ContainerPlanMetrics(
        containers=int(containers),
        total_cbm_used=float(total_used),
        total_cbm_capacity=float(total_capacity),
        avg_utilization=float(avg_util),
        weighted_utilization=float(weighted_util),
        low_util_count=int(low_util_count),
        low_util_threshold=float(low_util_threshold),
        ape_vs_planned=float(ape_vs_planned),
        ape_vs_base=float(0),
        ape_vs_excess=float(0),
        overall_score=float(overall),
        total_cbm_used_by_container_dest=total_cbm_used_by_container_dest,
        total_excess_in_cbm_by_container=total_excess_in_cbm_by_container,
        container_utilization_status_info=container_utilization_status_info,
        demand_met_by_sku=demand_met_by_sku_list,
    )

    vendor.container_plans[-1].metrics = latest_plan_metrics    

    print("-------After Move-----------")
    print(pd.DataFrame(latest_plan_metrics.total_cbm_used_by_container_dest))    
    print("--------------------------------")
    
    print("-------Demand Met by SKU (in eaches)-----------")
    if demand_met_by_sku_list:
        demand_df = pd.DataFrame(demand_met_by_sku_list)
        # Sort by delta to show biggest differences first
        demand_df = demand_df.sort_values(by="delta", ascending=False)
        print(demand_df[demand_df["delta"] != 0].to_string(index=False))
    else:
        print("No SKU demand Delta values available")
    print("--------------------------------")

    return vendor

def plan_eval_router(vendor: vendorState) -> str:
    if not vendor.container_plans:
        return "end"

    plan = vendor.container_plans[-1]
    move_action = str(getattr(getattr(plan, "moveProposal", None), "action", "") or "").lower()
    is_do_nothing_move = move_action == "do_nothing"
    loop_counter = getattr(plan, "plan_loop_counter", 1) or 1

    over_loop_limit = loop_counter > PLAN_EVAL_MAX_LOOPS
    is_last_strategy = (plan.strategy == _STRATEGY_ORDER[-1])
    is_first_strategy = plan.strategy == PlanStrategy.BASE_PLAN

    if is_first_strategy:
        return "next_plan"
    elif is_do_nothing_move and is_last_strategy:
        return "end"
    elif over_loop_limit and not is_last_strategy:
        return "next_plan"
    elif over_loop_limit and is_last_strategy:
        return "end"
    elif not is_do_nothing_move:
        plan.plan_loop_counter += 1
        return "next_move"
    elif is_do_nothing_move:
        return "next_plan"
    else: #should not happen
        return "end"
