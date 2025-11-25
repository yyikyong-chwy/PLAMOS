from __future__ import annotations
from typing import Dict, List, Optional
from math import floor
from collections import defaultdict
import pandas as pd

from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerRow import ContainerPlanRow


# ---------------------------- helpers (shared) ---------------------------- #

def _safe_get(obj, attr_path: str, default=None):
    """
    Safely get nested attributes/keys; supports dicts and objects.
    Example: _safe_get(move, "reduce.cbm_goal", 0.0)
    """
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


# ---------------------------- move: CONSOLIDATE ---------------------------- #

def consolidate_move(vendor: vendorState, plan: ContainerPlanState, move: Dict) -> None:
    """
    Move as much as possible from src -> dst, respecting MCP multiples and CBM_Max.
    """
    CBM_Max: float = float(getattr(vendor, "CBM_Max", 66.0))

    src_cid = _safe_get(move, "consolidate.from_container", None)
    dst_cid = _safe_get(move, "consolidate.to_container", None)
    if src_cid is None or dst_cid is None:
        return  # malformed

    # Destination context
    dst_dest = _dest_for_container(plan, dst_cid)
    _ = _dest_for_container(plan, src_cid)  # src_dest (not needed currently)

    dst_cbm_now = _cbm_in_container(plan, dst_cid)
    dst_cbm_cap_left = max(0.0, CBM_Max - dst_cbm_now)
    if dst_cbm_cap_left <= 0:
        return  # already full

    # Source rows
    src_rows: List[ContainerPlanRow] = [r for r in plan.container_plan_rows if r.container == src_cid]

    # (optional debug)
    # plan.to_df().to_csv("plan_before_move.csv", index=False)

    # Greedy: move cases in MCP multiples until destination full or source empty
    for srow in src_rows:
        if dst_cbm_cap_left <= 0:
            break

        cases_src = int(srow.cases_assigned or 0)
        if cases_src <= 0:
            continue

        cbm_per_case = float(srow.case_pk_CBM or 0.0)
        if cbm_per_case <= 0.0:
            continue

        mcp = max(1, int(srow.master_case_pack or 1))

        # Fit by destination capacity first
        max_cases_by_dst_cap = floor(dst_cbm_cap_left / cbm_per_case)
        if max_cases_by_dst_cap <= 0:
            break

        # Also bound by how many cases the source row has
        max_cases_possible = min(cases_src, max_cases_by_dst_cap)

        # Round down to MCP multiple
        #cases_to_move = (max_cases_possible // mcp) * mcp #this is a stupid bug!!!!!!!!!!
        cases_to_move = max_cases_possible # it is already in cases! no need to round down to MCP multiple
        if cases_to_move <= 0:
            continue

        cbm_to_move = cases_to_move * cbm_per_case
        # (Safety) ensure we never exceed cap due to float precision
        if (dst_cbm_now + cbm_to_move) - CBM_Max > 1e-9:
            fit_by_cap = floor((CBM_Max - dst_cbm_now) / cbm_per_case)
            cases_to_move = (fit_by_cap // mcp) * mcp
            if cases_to_move <= 0:
                continue
            cbm_to_move = cases_to_move * cbm_per_case

        # Move from source
        srow.cases_assigned = int(srow.cases_assigned) - cases_to_move
        srow.cbm_assigned   = float(srow.cbm_assigned or 0.0) - cbm_to_move
        if srow.cases_assigned < 0: srow.cases_assigned = 0
        if srow.cbm_assigned < 0:   srow.cbm_assigned = 0.0

        # Move to destination
        drow = _get_or_create_dst_row(plan, srow, dst_cid, dst_dest)
        drow.cases_assigned = int(drow.cases_assigned or 0) + cases_to_move
        drow.cbm_assigned   = float(drow.cbm_assigned or 0.0) + cbm_to_move

        # Update destination fill
        dst_cbm_now      += cbm_to_move
        dst_cbm_cap_left  = max(0.0, CBM_Max - dst_cbm_now)


# ---------------------------- move: REDUCE ---------------------------- #
def reduce_move(vendor: vendorState, plan: ContainerPlanState, move: Dict) -> None:
    """
    Two-tier reduction strategy:
      Tier 1: Reduce up to excess_demand, prioritizing low F90_DAILY_AVG SKUs.
      Tier 2: If cbm_goal remains, reduce further but never below baseDemand.
    """
    #this is useless since llm tends to hallucinate the cbm_goal
    #cbm_goal = float(_safe_get(move, "reduce.cbm_goal", 0.0) or 0.0)    
    container = int(_safe_get(move, "reduce.container", 0) or 0)
    cbm_goal = _cbm_in_container(plan, container)

    plan_rows: List[ContainerPlanRow] = plan.container_plan_rows
    if not plan_rows:
        return

    # Map of SKU -> plan rows
    rows_by_sku: Dict[str, List[ContainerPlanRow]] = defaultdict(list)
    for r in plan_rows:
        rows_by_sku[str(r.product_part_number)].append(r)

    # Helper to compute total assigned CBM for a SKU
    def total_assigned_cbm(sku: str) -> float:
        return sum(float(r.cbm_assigned or 0.0) for r in rows_by_sku.get(sku, []))

    cbm_removed = 0.0
    goal_left = cbm_goal

    # --- Tier 1: Reduce from excess ---
    candidates_t1 = []
    for s in getattr(vendor, "ChewySku_info", []):
        sku = str(s.product_part_number)
        mcp = int(s.MCP or 0)
        cbm_case = float(s.case_pk_CBM or 0.0)
        excess_eaches = float(s.excess_demand or 0.0)
        f90 = float(s.F90_DAILY_AVG or 0.0)
        if mcp <= 0 or cbm_case <= 0 or excess_eaches <= 0:
            continue
        removable_cases = int(excess_eaches // mcp)
        if removable_cases > 0 and sku in rows_by_sku:
            candidates_t1.append({
                "sku": sku,
                "f90": f90,
                "cbm_case": cbm_case,
                "max_cases": removable_cases,
            })

    candidates_t1.sort(key=lambda x: x["f90"])

    for c in candidates_t1:
        if goal_left <= 0.01:
            break

        sku = c["sku"]
        cbm_case = c["cbm_case"]
        removable_cases = c["max_cases"]
        rows = sorted(rows_by_sku[sku], key=lambda r: float(r.case_pk_CBM or 0.0), reverse=True)

        for row in rows:
            if goal_left <= 0.01 or removable_cases <= 0:
                break
            cases_to_remove = min(removable_cases, int(row.cases_assigned or 0))
            cbm_delta = cases_to_remove * cbm_case
            if cbm_delta > goal_left:
                cases_to_remove = int(goal_left // cbm_case)
                cbm_delta = cases_to_remove * cbm_case
            if cases_to_remove <= 0:
                continue

            # Apply removal
            row.cases_assigned = max(0, int(row.cases_assigned or 0) - cases_to_remove)
            row.cbm_assigned = row.cases_assigned * cbm_case
            cbm_removed += cbm_delta
            goal_left = cbm_goal - cbm_removed
            removable_cases -= cases_to_remove

    # --- Tier 2: Reduce but not below baseDemand ---
    if goal_left > 0.01:
        candidates_t2 = []
        for s in getattr(vendor, "ChewySku_info", []):
            sku = str(s.product_part_number)
            mcp = int(s.MCP or 0)
            cbm_case = float(s.case_pk_CBM or 0.0)
            base_eaches = float(s.baseDemand or 0.0)
            f90 = float(s.F90_DAILY_AVG or 0.0)
            if mcp <= 0 or cbm_case <= 0:
                continue
            base_cases = int(base_eaches // mcp)
            assigned_cases = sum(int(r.cases_assigned or 0) for r in rows_by_sku.get(sku, []))
            removable_cases = max(0, assigned_cases - base_cases)
            if removable_cases > 0 and sku in rows_by_sku:
                candidates_t2.append({
                    "sku": sku,
                    "f90": f90,
                    "cbm_case": cbm_case,
                    "max_cases": removable_cases,
                })

        candidates_t2.sort(key=lambda x: x["f90"])

        for c in candidates_t2:
            if goal_left <= 1e-9:
                break

            sku = c["sku"]
            cbm_case = c["cbm_case"]
            removable_cases = c["max_cases"]
            rows = sorted(rows_by_sku[sku], key=lambda r: float(r.case_pk_CBM or 0.0), reverse=True)

            for row in rows:
                if goal_left <= 1e-9 or removable_cases <= 0:
                    break
                cases_to_remove = min(removable_cases, int(row.cases_assigned or 0))
                cbm_delta = cases_to_remove * cbm_case
                if cbm_delta > goal_left:
                    cases_to_remove = int(goal_left // cbm_case)
                    cbm_delta = cases_to_remove * cbm_case
                if cases_to_remove <= 0:
                    continue

                row.cases_assigned = max(0, int(row.cases_assigned or 0) - cases_to_remove)
                row.cbm_assigned = row.cases_assigned * cbm_case
                cbm_removed += cbm_delta
                goal_left = cbm_goal - cbm_removed
                removable_cases -= cases_to_remove

     # --- Tier 3: Reduce proportionally from remaining SKUs until goal met ---
    if goal_left > 0.01:
        all_rows = [r for r in plan_rows if (r.cbm_assigned or 0.0) > 0]
        total_cbm_now = sum(float(r.cbm_assigned or 0.0) for r in all_rows)
        if total_cbm_now > 0:
            reduction_ratio = min(1.0, goal_left / total_cbm_now)
            for r in all_rows:
                cbm_to_remove = float(r.cbm_assigned or 0.0) * reduction_ratio
                cbm_case = float(r.case_pk_CBM or 0.0)
                if cbm_case <= 0:
                    continue
                cases_to_remove = int(cbm_to_remove // cbm_case)
                if cases_to_remove <= 0:
                    continue
                r.cases_assigned = max(0, int(r.cases_assigned or 0) - cases_to_remove)
                r.cbm_assigned = r.cases_assigned * cbm_case
                cbm_removed += cbm_case * cases_to_remove
                goal_left = max(0.0, cbm_goal - cbm_removed)
                if goal_left <= 0.01:
                    break
    # Done – goal achieved or exhausted feasible removals
    return




# ---------------------------- move: PAD ---------------------------- #
def pad_move(vendor: vendorState, plan: ContainerPlanState, move: Dict) -> None:
    """
    Pad a specific underutilized container up to CBM_Max (never exceed).
    Candidate SKUs come from vendor.ChewySkuState, prioritizing:
      1) SKUs with no excess_demand (<= 0); fallback to all SKUs
      2) Sort by t90*mcp*cbm_case (desc)
    Enhancement: distribute remaining_cbm across TOP-3 candidates (weighted), not greedy.
    Respects MCP multiples and updates/creates rows in the target container.
    """
    # --- identify target container & capacity context ---
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

    # --- small helpers (local) ---
    from collections import defaultdict
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

        # If SKU exists elsewhere in the plan, clone its shape for the target container
        proto = _any_row_for_sku(sku)
        if proto is not None:
            return _get_or_create_dst_row(plan, proto, dst_cid, dst_dest)

        # Otherwise build a minimal, valid new row
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

    # --- build and sort candidate pool ---
    no_excess: List[Dict] = []
    all_skus: List[Dict] = []
    for s in getattr(vendor, "ChewySku_info", []) or []:
        try:
            sku = str(s.product_part_number)
            t90 = float(s.T90_DAILY_AVG or 0.0)
            mcp = int(s.MCP or 0)
            cbm_case = float(s.case_pk_CBM or 0.0)
            excess = float(s.excess_demand or 0.0)
        except Exception:
            continue
        if mcp <= 0 or cbm_case <= 0.0:
            continue
        rec = {"sku": sku, "t90": t90, "mcp": mcp, "cbm_case": cbm_case}
        all_skus.append(rec)
        if excess <= 0.0:
            no_excess.append(rec)

    pool = no_excess if no_excess else all_skus
    # Expand if too small or weak runrates
    if len(pool) < 10 or all(x["t90"] < 10 for x in pool):
        pool = all_skus
    if not pool:
        return

    # Sort by product score (desc): fast movers / bigger volume / larger cbm
    def score(x: Dict) -> float:
        return float(x["t90"]) * float(x["mcp"]) * float(x["cbm_case"])
    pool.sort(key=lambda x: -score(x))

    # ---- NEW: distribute across TOP-N (N=3) candidates by weight ----
    TOP_N = min(3, len(pool))
    top = pool[:TOP_N]

    # Weights from the same score used for ranking (avoid zero-division)
    weights = [max(1e-9, score(c)) for c in top]
    total_w = sum(weights)
    cbm_left = remaining_cbm

    # 1st pass: proportional allocation by weight, snapped to MCP multiples
    for i, c in enumerate(top):
        if cbm_left <= 0.1:
            break
        sku, mcp, cbm_case = c["sku"], int(c["mcp"]), float(c["cbm_case"])
        w = weights[i] / total_w
        target_cbm = cbm_left * w

        # Max cases by per-sku target and remaining capacity (conservative)
        max_cases_by_target = int(target_cbm // cbm_case)
        max_cases_by_cap    = int(cbm_left   // cbm_case)
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

    # 2nd pass (optional): round-robin fill leftover among top-N (one MCP chunk at a time)
    # This squeezes remaining capacity that couldn't be allocated due to rounding in pass 1.
    made_progress = True
    while cbm_left >= 1e-6 and made_progress:
        made_progress = False
        for c in top:
            if cbm_left < 1e-6:
                break
            sku, mcp, cbm_case = c["sku"], int(c["mcp"]), float(c["cbm_case"])

            # At least one MCP chunk must fit
            mcp_chunk_cbm = mcp * cbm_case
            if mcp_chunk_cbm <= cbm_left + 1e-9:
                drow = _ensure_row_for_sku_in_container(sku, cbm_case, mcp)
                drow.cases_assigned = int(drow.cases_assigned or 0) + mcp
                drow.cbm_assigned = float(drow.cases_assigned) * float(drow.case_pk_CBM or cbm_case)
                cbm_left = max(0.0, cbm_left - mcp_chunk_cbm)
                made_progress = True


# ---------------------------- main entrypoint ---------------------------- #

def planMoveExecutorAgent(vendor: vendorState) -> vendorState:
    """
    Deterministically executes a single planner move proposal in vendor.container_plans[-1].moveProposal.
    Dispatches to per-move functions and performs common cleanup.
    """
    if not vendor.container_plans:
        return vendor

    plan: ContainerPlanState = vendor.container_plans[-1]
    move: Dict = getattr(plan, "moveProposal", None) or {}
    if not move:
        return vendor

    mv_type = str(_safe_get(move, "action", "") or "").lower()

    if mv_type == "consolidate":
        consolidate_move(vendor, plan, move)
    elif mv_type == "reduce":
        reduce_move(vendor, plan, move)
    elif mv_type == "pad":
        pad_move(vendor, plan, move)
    else:
        # Unknown action: no-op
        return vendor

    # Common cleanup for all moves
    plan.container_plan_rows = [
        r for r in plan.container_plan_rows if int(r.cases_assigned or 0) > 0
    ]


    # (optional debug snapshots)
    # plan.to_df().to_csv("plan_after_move.csv", index=False)

    return vendor
