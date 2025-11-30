# requirements:
#   pip install langgraph pydantic pandas

from __future__ import annotations
from typing import Dict, List, Tuple, Any, TypedDict
from collections import defaultdict
import pandas as pd

# --- your state classes (import from your project) ---
# ContainerPlanRow: fields include vendor_Code, vendor_name, DEST, container, product_part_number,
#                   master_case_pack, case_pk_CBM, cases_needed, cases_assigned, cbm_assigned
from states.ContainerRow import ContainerPlanRow  # :contentReference[oaicite:5]{index=5}
from states.vendorState import vendorState                 # :contentReference[oaicite:7]{index=7}


from typing import Any, List, Tuple
from collections import defaultdict
import pandas as pd

# ------------- helper: FFD container assignment for a single vendor -------------
def assign_containers_for_df(
    df_plan: pd.DataFrame,
    cbm_Max: float,
    group_cols: Tuple[str, ...] = ("vendor_Code", "DEST"),
    *,
    vendor_col: str = "vendor_Code",
) -> pd.DataFrame:
    """
    Assign containers using First-Fit Decreasing (FFD) for a SINGLE vendor to multiple DESTs.

    Required columns in df_plan:
      vendor_Code, DEST, product_part_number, master_case_pack, case_pk_CBM, cases_needed

    Args:
        df_plan: Container plan rows for ONE vendor across DESTs.
        cbm_Max: Container capacity (CBM) for this vendor.
        group_cols: Grouping (kept so numbering is per vendor, filling per (vendor, DEST)).
        vendor_col: Column holding the vendor code.

    Returns:
        DataFrame with original columns plus:
          - cases_assigned
          - cbm_assigned
          - container  (container ids increment per vendor across DESTs)
    """
    # Validate required columns
    req = ["vendor_Code", "DEST", "product_part_number", "master_case_pack", "case_pk_CBM", "cases_needed"]
    missing = [c for c in req if c not in df_plan.columns]
    if missing:
        raise KeyError(f"Container plan is missing required columns: {missing}")

    # Working copy + dtypes
    df = df_plan.copy()
    df["master_case_pack"] = pd.to_numeric(df["master_case_pack"], errors="coerce").fillna(0).astype(int)
    df["case_pk_CBM"]     = pd.to_numeric(df["case_pk_CBM"], errors="coerce").fillna(0.0)
    df["cases_needed"]    = pd.to_numeric(df["cases_needed"], errors="coerce").fillna(0).astype(int)

    # Sort by group then decreasing per-case CBM for deterministic FFD
    sort_cols = list(group_cols) + ["case_pk_CBM"]
    df = df.sort_values(sort_cols, ascending=[True]*len(group_cols) + [False])

    out_rows: List[dict] = []

    # Container id counter PER VENDOR (ids increment across all DESTs for this vendor)
    vendor_counter = defaultdict(int)  # vendor_code -> last container id

    for gkey, g in df.groupby(list(group_cols), sort=False):
        sample = g.iloc[0]
        vkey = str(sample[vendor_col]).strip() if pd.notna(sample[vendor_col]) else None

        # Containers are per (vendor, DEST) group, but ids are per vendor
        containers = []  # each: {'cbm_used': float, 'id': int}

        def new_container_id_for_vendor(vendor_key: Any) -> int:
            vendor_counter[vendor_key] += 1
            return vendor_counter[vendor_key]

        def open_new_container():
            cid = new_container_id_for_vendor(vkey)
            containers.append({"cbm_used": 0.0, "id": cid})
            return cid

        # Iterate SKUs within the (vendor, DEST) group
        for _, row in g.iterrows():
            cbm_case = float(row["case_pk_CBM"])
            mcp      = int(row["master_case_pack"])
            rem_cases = int(row["cases_needed"])

            # Sanity checks
            if mcp <= 0 or cbm_case <= 0.0 or rem_cases <= 0:
                continue

            # If a single case can't fit in an empty container, skip this SKU
            if cbm_case > cbm_Max:
                continue

            while rem_cases > 0:
                best_idx = None
                best_free_after = None
                best_fit_cases = 0

                # Try to place into existing containers first
                for idx, cont in enumerate(containers):
                    free_hard = cbm_Max - cont["cbm_used"]
                    if free_hard <= 0:
                        continue
                    fit_hard = int(free_hard // cbm_case)  # integer cases that fit
                    if fit_hard >= 1:
                        assign = min(rem_cases, fit_hard)
                        free_after = free_hard - assign * cbm_case
                        if best_idx is None or free_after < best_free_after:
                            best_idx = idx
                            best_free_after = free_after
                            best_fit_cases = assign

                if best_idx is None:
                    # Open a new container for this vendor (global id), dedicated to this DEST group
                    cid = open_new_container()
                    cont = containers[-1]
                    free_hard = cbm_Max - cont["cbm_used"]
                    fit_hard = int(free_hard // cbm_case)
                    assign = min(rem_cases, max(fit_hard, 0))
                    if assign == 0:
                        # Shouldn't happen (cbm_case <= capacity), but guard anyway
                        break
                else:
                    assign = best_fit_cases
                    cid = containers[best_idx]["id"]

                cbm_assigned = assign * cbm_case

                # Update the container's CBM usage
                if best_idx is None:
                    containers[-1]["cbm_used"] += cbm_assigned
                else:
                    containers[best_idx]["cbm_used"] += cbm_assigned

                rem_cases -= assign

                out = row.to_dict()
                out.update({
                    "cases_assigned": int(assign),
                    "cbm_assigned": float(cbm_assigned),
                    "container": int(cid),
                })
                out_rows.append(out)

    if not out_rows:
        cols = list(df_plan.columns) + ["cases_assigned", "cbm_assigned", "container"]
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(out_rows)
    #print("\n\nResult:\n", result.columns)

    return result


def write_back_assignments(vendor: vendorState, assigned_df: pd.DataFrame) -> vendorState:
    """
    Overwrite the first container plan's rows with assigned values for:
      cases_assigned, cbm_assigned, container

    Matching key: (vendor_Code, DEST, product_part_number)
    """
    if not vendor.container_plans:
        return

    # Keep only fields that exist on ContainerPlanRow
    row_fields = set(ContainerPlanRow.model_json_schema()["properties"].keys())


    records: List[ContainerPlanRow] = []

    for rec in assigned_df.to_dict(orient="records"):
        payload = {k: rec.get(k) for k in row_fields}
        # Pydantic v2: model_validate (or use ContainerPlanRow(**payload))
        records.append(ContainerPlanRow.model_validate(payload))

    vendor.container_plans[0].container_plan_rows = records
    return vendor




def basePlanAgent(state: vendorState) -> vendorState:

    cbm_Max = state.CBM_Max

    #obtain the base plan for this vendor
    df_plan = state.container_plans[0].to_df() 


    assigned = assign_containers_for_df(
            df_plan,
            cbm_Max=cbm_Max,
            group_cols=("vendor_Code", "DEST"),
            vendor_col="vendor_Code",
        )
    state = write_back_assignments(state, assigned)

    return state

