from __future__ import annotations
from typing import Dict, List, Optional
from math import floor
from collections import defaultdict

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
        cases_to_move = (max_cases_possible // mcp) * mcp
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
    Remove CBM up to cbm_goal by trimming excess_demand on low-runrate SKUs.
    - Prioritize lower T90_DAILY_AVG SKUs first; within SKU, drain rows with higher cbm_per_case first.
    - If total feasible CBM from excess < goal, abort without changes.
    """
    cbm_goal = float(_safe_get(move, "reduce.cbm_goal", 0.0) or 0.0)
    if cbm_goal <= 0.0:
        return

    plan_rows: List[ContainerPlanRow] = plan.container_plan_rows
    if not plan_rows:
        return

    # Assigned cases & row buckets by SKU
    assigned_by_sku = defaultdict(int)
    rows_by_sku: Dict[str, List[ContainerPlanRow]] = defaultdict(list)
    for r in plan_rows:
        sku = str(r.product_part_number)
        assigned_by_sku[sku] += int(r.cases_assigned or 0)
        rows_by_sku[sku].append(r)

    # Collect feasible candidates from vendor's ChewySku_info
    candidates = []
    for s in getattr(vendor, "ChewySku_info", []) or []:
        try:
            sku = str(s.product_part_number)
            mcp = int(s.MCP or 0)
            cbm_per_case = float(s.case_pk_CBM or 0.0)
            excess_eaches = float(s.excess_demand or 0.0)
            t90 = float(s.T90_DAILY_AVG or 0.0)
        except Exception:
            continue

        if mcp <= 0 or cbm_per_case <= 0.0 or excess_eaches <= 0.0:
            continue

        removable_cases_from_excess = int(excess_eaches // mcp)
        if removable_cases_from_excess <= 0:
            continue

        total_assigned_cases_for_sku = int(assigned_by_sku.get(sku, 0))
        if total_assigned_cases_for_sku <= 0:
            continue

        max_removable_cases = min(removable_cases_from_excess, total_assigned_cases_for_sku)
        if max_removable_cases <= 0:
            continue

        candidates.append({
            "sku": sku,
            "t90": t90,
            "cbm_per_case": cbm_per_case,
            "max_cases": max_removable_cases,
        })

    if not candidates:
        return  # nothing to trim

    # Pre-check feasibility: can we hit cbm_goal?
    total_possible_cbm = sum(c["max_cases"] * c["cbm_per_case"] for c in candidates)
    if total_possible_cbm + 1e-9 < cbm_goal:
        # Not enough excess -> abort w/ NO changes
        return

    # Execute: lowest T90 first, then higher CBM/Case rows first
    candidates.sort(key=lambda x: (x["t90"], -x["cbm_per_case"]))
    cbm_removed = 0.0
    goal_left = cbm_goal

    for c in candidates:
        if goal_left <= 1e-9:
            break

        sku = c["sku"]
        cbm_case = c["cbm_per_case"]
        cases_left_for_sku = int(min(c["max_cases"], assigned_by_sku.get(sku, 0)))
        if cases_left_for_sku <= 0:
            continue

        sku_rows = rows_by_sku.get(sku, [])
        sku_rows_sorted = sorted(sku_rows, key=lambda r: float(r.case_pk_CBM or 0.0), reverse=True)

        for row in sku_rows_sorted:
            if goal_left <= 1e-9 or cases_left_for_sku <= 0:
                break

            row_cases = int(row.cases_assigned or 0)
            if row_cases <= 0:
                continue

            max_by_row = min(row_cases, cases_left_for_sku)

            # Cases needed by remaining goal (integer)
            max_by_goal = int(goal_left // cbm_case)
            if max_by_goal <= 0 and goal_left > 1e-12:
                cases_to_remove = 1 if max_by_row >= 1 else 0
            else:
                cases_to_remove = min(max_by_row, max_by_goal)

            if cases_to_remove <= 0:
                continue

            cbm_delta = cases_to_remove * cbm_case

            # Apply to row
            row.cases_assigned = int(row.cases_assigned) - cases_to_remove
            if row.cases_assigned < 0:
                row.cases_assigned = 0
            row.cbm_assigned = float(row.cases_assigned) * float(row.case_pk_CBM or 0.0)

            # Track
            assigned_by_sku[sku] -= cases_to_remove
            cases_left_for_sku  -= cases_to_remove
            cbm_removed         += cbm_delta
            goal_left           = max(0.0, cbm_goal - cbm_removed)

    # Feasibility check guarantees we should be at or above goal within epsilon.


# ---------------------------- move: PAD (stub) ---------------------------- #

# ---------------------------- move: PAD ---------------------------- #

def pad_move(vendor: vendorState, plan: ContainerPlanState, move: Dict) -> None:
    """
    Increase order to hit cbm_goal by padding high-runrate SKUs that do NOT have excess_demand.
    If none exist, fall back to all SKUs (still prioritize higher T90_DAILY_AVG).
    - Assigns pads into the specified container.
    - Respects MCP multiples.
    - Two-pass strategy: floor-fill first (no overshoot), then minimally overshoot with one SKU if needed.
    """
    # ---- extract inputs ----
    cbm_goal = float(_safe_get(move, "pad.cbm_goal", 0.0) or 0.0)
    if cbm_goal <= 0.0:
        return

    dst_cid = _safe_get(move, "pad.container", None)
    if dst_cid is None:
        dst_cid = _safe_get(move, "pad.to_container", None)
    if dst_cid is None:
        return  # malformed: no target container

    dst_dest = _dest_for_container(plan, int(dst_cid))
    if dst_dest is None:
        # If the plan has no rows for this container yet, we cannot infer DEST.
        # You can set a default or abort. We'll abort for safety.
        return

    # ---- helpers ----
    from collections import defaultdict
    rows_by_sku: Dict[str, List[ContainerPlanRow]] = defaultdict(list)
    for r in plan.container_plan_rows:
        rows_by_sku[str(r.product_part_number)].append(r)

    def _any_row_for_sku(sku: str) -> Optional[ContainerPlanRow]:
        lst = rows_by_sku.get(sku, [])
        return lst[0] if lst else None

    def _ensure_row_for_sku_in_container(sku: str, cbm_per_case: float, mcp: int) -> ContainerPlanRow:
        """
        Ensure there's a row for (sku, dst_cid). Prefer cloning an existing row (proto) for the SKU.
        If none exist in the plan, synthesize one from ChewySkuState.
        """
        # 1) If there's already a row in target container, return it
        for r in plan.container_plan_rows:
            if r.container == dst_cid and str(r.product_part_number) == sku:
                # Ensure DEST consistency
                if getattr(r, "DEST", None) != dst_dest:
                    setattr(r, "DEST", dst_dest)
                return r

        # 2) If there is any row for this SKU elsewhere, clone its shape as proto
        proto = _any_row_for_sku(sku)
        if proto is not None:
            return _get_or_create_dst_row(plan, proto, dst_cid, dst_dest)

        # 3) Build from ChewySkuState (minimal fields)
        #    We keep it resilient to extra/unknown fields by using model_validate.
        sku_state = None
        for s in getattr(vendor, "ChewySku_info", []) or []:
            if str(getattr(s, "product_part_number", "")) == sku:
                sku_state = s
                break

        base = {}
        if sku_state is not None:
            base = {
                "product_part_number": sku,
                "master_case_pack": int(getattr(sku_state, "MCP", mcp) or mcp or 1),
                "case_pk_CBM": float(getattr(sku_state, "case_pk_CBM", cbm_per_case) or cbm_per_case or 0.0),
                "vendor_Code": getattr(vendor, "vendor_Code", None),
                "product_name": getattr(sku_state, "product_name", None),
            }
        else:
            base = {
                "product_part_number": sku,
                "master_case_pack": int(mcp or 1),
                "case_pk_CBM": float(cbm_per_case or 0.0),
                "vendor_Code": getattr(vendor, "vendor_Code", None),
            }

        base.update({
            "container": int(dst_cid),
            "DEST": dst_dest,
            "cases_assigned": 0,
            "cbm_assigned": 0.0,
        })
        new_r = ContainerPlanRow.model_validate(base)
        plan.container_plan_rows.append(new_r)
        rows_by_sku[sku].append(new_r)
        return new_r

    # ---- build candidate SKU list (prefer no-excess, high run rate) ----
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

    # Priority pool: SKUs without excess first; if none, use all
    pool = no_excess if no_excess else all_skus

    # Sort: higher run rate first; then prefer smaller MCP (finer granularity); then larger cbm/case
    pool.sort(key=lambda x: (-x["t90"], x["mcp"], -x["cbm_case"]))

    if not pool:
        return

    # ---- PASS 1: floor-fill without overshoot ----
    cbm_added = 0.0
    goal_left = cbm_goal

    for c in pool:
        if goal_left <= 1e-9:
            break
        sku, mcp, cbm_case = c["sku"], int(c["mcp"]), float(c["cbm_case"])

        # how many cases can we add without exceeding the remaining goal?
        max_cases_by_goal = int(goal_left // cbm_case)  # floor
        cases_to_add = (max_cases_by_goal // mcp) * mcp
        if cases_to_add <= 0:
            continue

        drow = _ensure_row_for_sku_in_container(sku, cbm_case, mcp)
        drow.cases_assigned = int(drow.cases_assigned or 0) + cases_to_add
        drow.cbm_assigned   = float(drow.cases_assigned) * float(drow.case_pk_CBM or cbm_case)

        delta = cases_to_add * cbm_case
        cbm_added += delta
        goal_left  = max(0.0, cbm_goal - cbm_added)

    # ---- PASS 2: minimal overshoot if still short ----
    if goal_left > 1e-9:
        # choose the SKU with smallest "step" needed to clear the remainder:
        # compute cases_needed rounded up to MCP multiples; pick the SKU that yields the minimal CBM overshoot
        best_choice = None
        best_over = None

        for c in pool:
            sku, mcp, cbm_case = c["sku"], int(c["mcp"]), float(c["cbm_case"])
            if cbm_case <= 0.0 or mcp <= 0:
                continue
            cases_needed = int(-(-goal_left // cbm_case))  # ceil(goal_left / cbm_case)
            # round UP to MCP multiple
            rounded = ((cases_needed + mcp - 1) // mcp) * mcp
            added_cbm = rounded * cbm_case
            overshoot = added_cbm - goal_left
            # track the smallest overshoot; tie-break by higher run rate then smaller MCP
            key = (overshoot, -c["t90"], c["mcp"])
            if best_over is None or key < best_over:
                best_over = key
                best_choice = (c, rounded, added_cbm)

        if best_choice is not None:
            c, cases_to_add, added_cbm = best_choice
            sku, mcp, cbm_case = c["sku"], int(c["mcp"]), float(c["cbm_case"])
            drow = _ensure_row_for_sku_in_container(sku, cbm_case, mcp)
            drow.cases_assigned = int(drow.cases_assigned or 0) + int(cases_to_add)
            drow.cbm_assigned   = float(drow.cases_assigned) * float(drow.case_pk_CBM or cbm_case)
            cbm_added += added_cbm
            goal_left  = max(0.0, cbm_goal - cbm_added)
    # done



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
