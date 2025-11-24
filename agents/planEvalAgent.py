
from __future__ import annotations
from typing import Dict, List, Tuple, Any, TypedDict
from collections import defaultdict
import pandas as pd
import numpy as np

# --- your state classes (import from your project) ---
# ContainerPlanRow: fields include vendor_Code, vendor_name, DEST, container, product_part_number,
#                   master_case_pack, case_pk_CBM, cases_needed, cases_assigned, cbm_assigned
from states.ContainerRow import ContainerPlanRow  # :contentReference[oaicite:5]{index=5}
from states.vendorState import vendorState                 # :contentReference[oaicite:7]{index=7}
from states.containerPlanState import ContainerPlanMetrics
from states.planStrategy import PlanStrategy

PLAN_EVAL_MAX_LOOPS = 10

#simple aid functions
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, None) else 0.0

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _sku_cases_map(vendor: vendorState) -> Dict[str, Dict[str, float]]:
    """
    Build a mapping for each SKU -> {MCP, planned_cases, base_cases, excess_cases}
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
            "base_cases":    to_cases(s.baseDemand),
            "excess_cases":  to_cases(s.excess_demand),
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

def planEvalAgent(vendor: vendorState) -> vendorState:

    if not vendor.container_plans:
        return vendor

    #weights to evaluate the plan, maybe arbitrary so lets change as it goes
    weights = {"util": 0.40, "planned": 0.30, "base": 0.20, "excess": 0.10}    
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
        target_df = pd.DataFrame(columns=["MCP", "planned_cases", "base_cases", "excess_cases"])
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
    for col in ("planned_cases", "base_cases", "excess_cases"):
        if col not in target_df.columns:
            target_df[col] = np.nan
    target_df = target_df.reindex(combined_index)

    # APEs (weighted)
    ape_vs_planned = _weighted_ape(final_cases, target_df["planned_cases"])
    ape_vs_base = _weighted_underfill_penalty(final_cases, target_df["base_cases"], down_weight=3.0)
    # For "excess": if excess target is 0, we ignore (weigh only positive targets)
    #need more thinking on this one
    #ape_vs_excess  = _weighted_ape(final_cases, target_df["excess_cases"])

    # Convert APE (error) to score
    planned_score = 1.0 - _clamp01(ape_vs_planned)
    base_score    = 1.0 - _clamp01(ape_vs_base)
    excess_score = 0.0 #defaulting to this for now
    #excess_score  = 1.0 - _clamp01(ape_vs_excess)

    # --- Overall score ---
    overall = (
        100.0
        * (weights.get("util", 0.40)   * util_score
         + weights.get("planned", 0.30)* planned_score
         + weights.get("base", 0.20)   * base_score
         + weights.get("excess", 0.10) * excess_score)
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
        # Unused = max(capacity - used, 0). If a container overflows, unused=0 (overflow can be inferred via cbm_used > capacity).
        used_by_cdest_df["unused_cbm"] = (used_by_cdest_df["capacity_cbm"] - used_by_cdest_df["cbm_used"]).clip(lower=0.0)

        total_cbm_used_by_container_dest = used_by_cdest_df.to_dict(orient="records")
    else:
        total_cbm_used_by_container_dest = [] #empty list if no df

    #    Allocate each SKU's excess_cases across its assigned containers
    #    proportional to cases_assigned per container, then convert to CBM.
    total_excess_in_cbm_by_container: Dict[Any, float] = {}
    if not df.empty and not target_df.empty:
        # per-SKU totals
        sku_total_cases = (
            df.groupby("product_part_number")["cases_assigned"].sum().astype(float)
        )
        # Merge row-level with sku totals and each row's case_pk_CBM
        rows = df[["product_part_number","container","cases_assigned","case_pk_CBM"]].copy()
        rows["product_part_number"] = rows["product_part_number"].astype(str)
        rows = rows.merge(
            sku_total_cases.rename("sku_cases_total").reset_index().rename(columns={"index":"product_part_number"}),
            on="product_part_number",
            how="left"
        )
        # Bring in SKU excess cases (in CASES)
        sku_excess_cases = target_df["excess_cases"].astype(float).fillna(0.0)
        rows = rows.merge(
            sku_excess_cases.rename("excess_cases").reset_index().rename(columns={"index":"product_part_number"}),
            on="product_part_number",
            how="left"
        )
        rows["sku_cases_total"] = rows["sku_cases_total"].fillna(0.0)
        rows["excess_cases"] = rows["excess_cases"].fillna(0.0)

        # Proportional allocation per row
        def _row_excess_cases(r):
            if r["sku_cases_total"] <= 0 or r["excess_cases"] <= 0:
                return 0.0
            share = r["cases_assigned"] / r["sku_cases_total"]
            return float(share * r["excess_cases"])

        rows["row_excess_cases"] = rows.apply(_row_excess_cases, axis=1)
        rows["row_excess_cbm"] = rows["row_excess_cases"] * rows["case_pk_CBM"].astype(float)

        by_container = rows.groupby("container")["row_excess_cbm"].sum().fillna(0.0)
        total_excess_in_cbm_by_container = {k: float(v) for k, v in by_container.to_dict().items()}

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
        ape_vs_base=float(ape_vs_base),
        ape_vs_excess=float(0),
        overall_score=float(overall),
        total_cbm_used_by_container_dest=total_cbm_used_by_container_dest,
        total_excess_in_cbm_by_container=total_excess_in_cbm_by_container,
    )

    vendor.container_plans[-1].metrics = latest_plan_metrics    

    print("-------After Move-----------")
    print(pd.DataFrame(latest_plan_metrics.total_cbm_used_by_container_dest))    
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
    is_last_strategy = plan.strategy == PlanStrategy.PAD_ONLY
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
