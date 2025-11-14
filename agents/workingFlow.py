import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

import data.demand_pull as demand_pull
import data.sql_lite_store as sql_lite_store

def process_and_locate_missing_records(df_demand, df_Keppler_Split_Perc, df_vendor_cbm):

    df_demand['product_part_number'] = df_demand['product_part_number'].astype(str)
    df_Keppler_Split_Perc['ITEM_ID'] = df_Keppler_Split_Perc['ITEM_ID'].astype(str)
    df_keppler_filtered = df_Keppler_Split_Perc[['ITEM_ID', 'MDT1_FRAC', 'TLA1_FRAC', 'TNY1_FRAC']].copy().rename(
    columns={'ITEM_ID':'CHW_SKU_NUMBER'})
    
    df_demand1 = df_demand[['product_part_number','Final Buy Qty']].copy().rename(
    columns={'product_part_number':'CHW_SKU_NUMBER', 'Final Buy Qty':'Demand'}) 

    df_demand2 = df_demand1.merge(
        df_keppler_filtered,
        how='left',               # left join keeps all rows from df_demand
        left_on='CHW_SKU_NUMBER',
        right_on='CHW_SKU_NUMBER'
    )

    df_demand3 = df_demand2.merge(
        df_vendor_cbm,
        how='left',
        on=['CHW_SKU_NUMBER'])

    df_demand3['MDT1_Demand'] = df_demand3['MDT1_FRAC'] * df_demand3['Demand']
    df_demand3['TLA1_Demand'] = df_demand3['TLA1_FRAC'] * df_demand3['Demand']
    df_demand3['TNY1_Demand'] = df_demand3['TNY1_FRAC'] * df_demand3['Demand']
    df_demand3 = snap_splits_to_mcp(df_demand3)

    #locating missing records
    df_demand_1 = df_demand1.merge(
        df_keppler_filtered,
        how='left',
        on='CHW_SKU_NUMBER',
        indicator=True  # adds a column called "_merge"
    )

    # Rows where no match was found on the right
    skus_w_no_splits = df_demand_1[df_demand_1['_merge'] == 'left_only']


    df_demand_2 = df_demand1.merge(
        df_vendor_cbm,
        how='left',
        on=['CHW_SKU_NUMBER'],
        indicator=True  # adds a column called "_merge"
    )

    # Rows where no match was found on the right
    skus_w_no_record_in_plm = df_demand_2[df_demand_2['_merge'] == 'left_only']
    

    df_demand4 = df_demand3.drop(columns=['Demand', 'MDT1_FRAC', 'TLA1_FRAC', 'TNY1_FRAC'], errors='ignore')
    # Melt the demand columns from wide to long format
    df_demand_long = df_demand4.melt(
        id_vars=[col for col in df_demand4.columns if not col.endswith('_Demand')],
        value_vars=['MDT1_Demand', 'TLA1_Demand', 'TNY1_Demand'],
        var_name='DEST',
        value_name='Demand'
    )

    # Clean up the DEST column to remove '_Demand' suffix
    df_demand_long['DEST'] = df_demand_long['DEST'].str.replace('_Demand', '')

    df_demand_long['CHW_MASTER_CASE_PACK'] = pd.to_numeric(df_demand_long['CHW_MASTER_CASE_PACK'], errors='coerce').fillna(0).astype(int)
    df_demand_long['CHW_MASTER_CARTON_CBM']     = pd.to_numeric(df_demand_long['CHW_MASTER_CARTON_CBM'], errors='coerce')
    df_demand_long['Demand']          = pd.to_numeric(df_demand_long['Demand'], errors='coerce').fillna(0.0)

    mask = df_demand_long['CHW_MASTER_CARTON_CBM'].notna() & (df_demand_long['CHW_MASTER_CASE_PACK'] > 0) & (df_demand_long['Demand'] > 0)
    mask2 = df_demand_long['CHW_MASTER_CARTON_CBM'].isna() | (df_demand_long['CHW_MASTER_CASE_PACK'].isna()) | (df_demand_long['Demand'].isna()) 
    mask3 = (df_demand_long['CHW_MASTER_CASE_PACK'] <= 0) | (df_demand_long['Demand'] < 0) | (df_demand_long['CHW_MASTER_CARTON_CBM'] <= 0) 

    excluded_skus_w_incomplete_data = df_demand_long.loc[mask2 | mask3].copy()
    df_demand_long = df_demand_long.loc[mask].reset_index(drop=True)


    df_demand_long['cases_needed'] = np.where(
            df_demand_long['CHW_MASTER_CASE_PACK'] > 0, np.ceil(df_demand_long['Demand'] / df_demand_long['CHW_MASTER_CASE_PACK']).astype(int), 0 )


    return df_demand_long, skus_w_no_splits, skus_w_no_record_in_plm, excluded_skus_w_incomplete_data


def snap_splits_to_mcp(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row:
      - Start from fractional splits (MDT1_FRAC, TLA1_FRAC, TNY1_FRAC)
      - Convert to cases using CHW_MASTER_CASE_PACK (MCP)
      - Floor cases per-destination
      - Distribute any remaining cases (up to floor(total_units/MCP)) by largest fractional remainder
      - Return per-destination unit demands snapped to MCP and NEVER exceeding the original Demand
    """
    frac_cols = ["MDT1_FRAC", "TLA1_FRAC", "TNY1_FRAC"]
    out_cols  = ["MDT1_Demand", "TLA1_Demand", "TNY1_Demand"]

    df = df.copy()
    for c in frac_cols:
        if c not in df.columns:
            df[c] = 0.0
    # normalize: if all three fractions are NA or 0, keep them 0s; otherwise normalize to sum=1
    fsum = df[frac_cols].sum(axis=1).replace(0, np.nan)
    for c in frac_cols:
        df[c] = df[c] / fsum
    df[frac_cols] = df[frac_cols].fillna(0.0)

    # Ensure numeric
    df["CHW_MASTER_CASE_PACK"] = pd.to_numeric(df["CHW_MASTER_CASE_PACK"], errors="coerce").fillna(0).astype(int)
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0)

    # Prepare outputs
    for c in out_cols:
        if c not in df.columns:
            df[c] = 0

    # Row-wise snapping
    def _snap_row(r):
        mcp = int(r.get("CHW_MASTER_CASE_PACK", 0))
        demand_units = float(r.get("Demand", 0.0))
        if mcp <= 0 or demand_units <= 0:
            return pd.Series({c: 0 for c in out_cols})

        # fractional suggested units
        target_units = {
            "MDT1": float(r["MDT1_FRAC"] * demand_units),
            "TLA1": float(r["TLA1_FRAC"] * demand_units),
            "TNY1": float(r["TNY1_FRAC"] * demand_units),
        }

        # convert to float cases
        target_cases_f = {k: v / mcp for k, v in target_units.items()}

        # initial floor of cases per dest
        cases_floor = {k: int(np.floor(vf)) for k, vf in target_cases_f.items()}

        # total cases allowed (not exceeding original demand)
        total_cases_cap = int(np.floor(demand_units / mcp))
        cases_sum = sum(cases_floor.values())
        cases_left = max(total_cases_cap - cases_sum, 0)

        if cases_left > 0:
            # remainders for fair distribution
            remainders = {k: (target_cases_f[k] - cases_floor[k]) for k in target_cases_f}
            # order by largest remainder (stable to keep input order if ties)
            order = sorted(remainders.keys(), key=lambda k: remainders[k], reverse=True)
            for k in order:
                if cases_left <= 0:
                    break
                cases_floor[k] += 1
                cases_left -= 1

        # final snapped units per dest (cases * mcp)
        snapped_units = {k: cases_floor[k] * mcp for k in cases_floor}

        return pd.Series({
            "MDT1_Demand": snapped_units["MDT1"],
            "TLA1_Demand": snapped_units["TLA1"],
            "TNY1_Demand": snapped_units["TNY1"],
        })

    snapped = df.apply(_snap_row, axis=1)
    df["MDT1_Demand"] = snapped["MDT1_Demand"].astype(int)
    df["TLA1_Demand"] = snapped["TLA1_Demand"].astype(int)
    df["TNY1_Demand"] = snapped["TNY1_Demand"].astype(int)

    # Optional diagnostics: how many units (if any) we could not allocate because of MCP snapping
    df["unallocated_units"] = (
        df["Demand"] - (df["MDT1_Demand"] + df["TLA1_Demand"] + df["TNY1_Demand"])
    )

    return df

def build_containers(
    df_demand: pd.DataFrame,
    capacity_map: Dict[Any, float],
    group_cols: Tuple[str, ...] = ('CHW_PRIMARY_SUPPLIER_NUMBER','DEST',),
    *,
    vendor_col: str = 'CHW_PRIMARY_SUPPLIER_NUMBER',
    default_capacity: float = 66.0):
    """
    Assign container numbers per group using FFD, with per-vendor container capacity.

    Parameters
    ----------
    df_demand : DataFrame
        Must contain:
          - 'Master Case Pack' (int)
          - 'Case Pk CBM' (float, CBM per case)
          - 'Demand' (float, units)
    capacity_map : dict
        {vendor -> CBM_Max} (e.g., {'V001': 68.0, 'V002': 72.5, ...})
    group_cols : tuple
        Grouping keys (by default ('Code','DEST'))
    vendor_col : str
        Column in df_long that identifies the vendor (default: 'Code')
    default_capacity : float
        Capacity to use if vendor not present in capacity_map

    Returns
    -------
    DataFrame
        Original columns +:
          - 'container' (1,2,3,... within each group)
          - 'cases_assigned'
          - 'cbm_assigned'
    """
    # --- Validate required cols
    req = ['CHW_MASTER_CASE_PACK', 'CHW_MASTER_CARTON_CBM', 'Demand']
    missing = [c for c in req if c not in df_demand.columns]
    if missing:
        raise KeyError(f"df_demand missing required columns: {missing}")

    # --- Safe numeric conversions
    df = df_demand.copy()     

    # --- FFD: sort by group, then decreasing per-case CBM
    sort_cols = list(group_cols) + ['CHW_MASTER_CARTON_CBM']
    df = df.sort_values(sort_cols, ascending=[True]*len(group_cols) + [False])

    out_rows = []

    # --- Process per group (e.g., per (Code, DEST))
    for gkey, g in df.groupby(list(group_cols), sort=False):
        # Resolve vendor for this group
        # (Assumes a single vendor per group; if group_cols doesn't include vendor_col,
        # we still pull vendor from rows below)
        sample_row = g.iloc[0]
        vendor_val = sample_row[vendor_col] if vendor_col in g.columns else sample_row.get(vendor_col, None)

        # Normalize & pick capacity
        vkey = str(vendor_val).strip() if vendor_val is not None else None
        try:
            group_capacity = float(capacity_map.get(vkey, default_capacity))
            if group_capacity <= 0:
                group_capacity = default_capacity
        except Exception:
            group_capacity = default_capacity

        # Track containers for this group
        containers = []  # each: {'cbm_used': float}
        def new_container():
            containers.append({'cbm_used': 0.0})
            return len(containers)  # 1-based id

        # Iterate SKUs
        for _, row in g.iterrows():
            cbm_case = float(row['CHW_MASTER_CARTON_CBM'])
            mcp      = int(row['CHW_MASTER_CASE_PACK'])
            rem_cases = int(row['cases_needed'])

            if mcp <= 0 or cbm_case <= 0:
                continue

            # If a single case can't fit in an empty container for this vendor, skip this SKU entirely
            if cbm_case > group_capacity:
                continue

            while rem_cases > 0:
                best_id = None
                best_free_after = None
                best_fit_cases = 0

                # Try to place into existing containers
                for cid, cont in enumerate(containers, start=1):
                    free_hard = group_capacity - cont['cbm_used']
                    if free_hard <= 0:
                        continue
                    fit_hard = int(free_hard // cbm_case)  # how many cases of this SKU can fit here
                    if fit_hard >= 1:
                        assign = min(rem_cases, fit_hard)
                        free_after = free_hard - assign * cbm_case
                        if best_id is None or free_after < best_free_after:
                            best_id = cid
                            best_free_after = free_after
                            best_fit_cases = assign

                if best_id is None:
                    # Open a new container and try again
                    best_id = new_container()
                    cont = containers[best_id - 1]
                    free_hard = group_capacity - cont['cbm_used']
                    fit_hard = int(free_hard // cbm_case)
                    assign = min(rem_cases, max(fit_hard, 0))

                    # If even an empty container can't fit a case, stop this SKU
                    if assign == 0:
                        break
                else:
                    assign = best_fit_cases

                # Record assignment
                cbm_assigned = assign * cbm_case
                containers[best_id - 1]['cbm_used'] += cbm_assigned
                rem_cases -= assign

                out = row.to_dict()
                out.update({
                    'cases_assigned': assign,
                    'cbm_assigned' : cbm_assigned,
                    'container'    : best_id,
                })
                out_rows.append(out)

    if not out_rows:
        return pd.DataFrame(columns=list(df_demand.columns) + ['cases_assigned','cbm_assigned','container'])

    result = pd.DataFrame(out_rows)

    # Keep original columns + new ones
    keep_cols = list(df_demand.columns) + ['cases_assigned', 'cbm_assigned', 'container']
    result = result[keep_cols]

    # Nice ordering (container after group keys; include product_part_number if present)
    order_front = list(group_cols) + ['container']
    if 'product_part_number' in result.columns and 'product_part_number' not in order_front:
        order_front += ['product_part_number']
    other_cols = [c for c in result.columns if c not in order_front]
    result = result[order_front + other_cols]

    return pd.DataFrame(result)


def build_containers_v2(
    df_demand: pd.DataFrame,
    capacity_map: Dict[Any, float],
    group_cols: Tuple[str, ...] = ('CHW_PRIMARY_SUPPLIER_NUMBER','DEST',),
    *,
    vendor_col: str = 'CHW_PRIMARY_SUPPLIER_NUMBER',
    default_capacity: float = 66.0):
    """
    Assign container numbers per group using FFD, with per-vendor container capacity.
    Container ids now increment per *vendor*, not per (vendor, DEST).
    """

    # --- Validate required cols
    req = ['CHW_MASTER_CASE_PACK', 'CHW_MASTER_CARTON_CBM', 'Demand', 'cases_needed']
    missing = [c for c in req if c not in df_demand.columns]
    if missing:
        raise KeyError(f"df_demand missing required columns: {missing}")

    # --- Sort by group, then decreasing per-case CBM for FFD determinism
    df = df_demand.copy()
    sort_cols = list(group_cols) + ['CHW_MASTER_CARTON_CBM']
    df = df.sort_values(sort_cols, ascending=[True]*len(group_cols) + [False])

    out_rows = []

    # --- Per-vendor global container id counter
    vendor_counter = defaultdict(int)  # vendor_key -> last_assigned_container_id

    # --- Process per (vendor, DEST) group; numbering is per vendor across groups
    for gkey, g in df.groupby(list(group_cols), sort=False):
        # Resolve vendor
        sample_row = g.iloc[0]
        vendor_val = sample_row[vendor_col] if vendor_col in g.columns else sample_row.get(vendor_col, None)
        vkey = str(vendor_val).strip() if vendor_val is not None else None

        # Capacity for this vendor
        try:
            group_capacity = float(capacity_map.get(vkey, default_capacity))
            if group_capacity <= 0:
                group_capacity = default_capacity
        except Exception:
            group_capacity = default_capacity

        # Track containers opened for THIS group only (contents do not cross DEST),
        # but assign ids from the per-vendor global counter.
        containers = []  # each: {'cbm_used': float, 'id': int}

        def new_container_id_for_vendor(vendor_key: Any) -> int:
            vendor_counter[vendor_key] += 1
            return vendor_counter[vendor_key]

        def new_container():
            cid = new_container_id_for_vendor(vkey)
            containers.append({'cbm_used': 0.0, 'id': cid})
            return cid  # global-per-vendor id

        # Iterate SKUs within the (vendor, DEST) group
        for _, row in g.iterrows():
            cbm_case = float(row['CHW_MASTER_CARTON_CBM'])
            mcp      = int(row['CHW_MASTER_CASE_PACK'])
            rem_cases = int(row['cases_needed'])

            if mcp <= 0 or cbm_case <= 0 or rem_cases <= 0:
                continue

            # If a single case can't fit in an empty container, skip this SKU
            if cbm_case > group_capacity:
                continue

            while rem_cases > 0:
                best_idx = None
                best_free_after = None
                best_fit_cases = 0

                # Try to place into existing containers (of this group)
                for idx, cont in enumerate(containers):
                    free_hard = group_capacity - cont['cbm_used']
                    if free_hard <= 0:
                        continue
                    fit_hard = int(free_hard // cbm_case)
                    if fit_hard >= 1:
                        assign = min(rem_cases, fit_hard)
                        free_after = free_hard - assign * cbm_case
                        if best_idx is None or free_after < best_free_after:
                            best_idx = idx
                            best_free_after = free_after
                            best_fit_cases = assign

                if best_idx is None:
                    # Open a new container for this vendor (global id), but it’s dedicated to this DEST group
                    cid = new_container()
                    cont = containers[-1]
                    free_hard = group_capacity - cont['cbm_used']
                    fit_hard = int(free_hard // cbm_case)
                    assign = min(rem_cases, max(fit_hard, 0))
                    if assign == 0:
                        break
                else:
                    assign = best_fit_cases
                    cid = containers[best_idx]['id']

                # Record assignment
                cbm_assigned = assign * cbm_case
                # Update the chosen container's cbm_used
                # (find by id; best_idx known if we chose existing)
                if best_idx is None:
                    containers[-1]['cbm_used'] += cbm_assigned
                else:
                    containers[best_idx]['cbm_used'] += cbm_assigned

                rem_cases -= assign

                out = row.to_dict()
                out.update({
                    'cases_assigned': assign,
                    'cbm_assigned' : cbm_assigned,
                    'container'    : cid,  # global per-vendor numbering
                })
                out_rows.append(out)

    if not out_rows:
        return pd.DataFrame(columns=list(df_demand.columns) + ['cases_assigned','cbm_assigned','container'])

    result = pd.DataFrame(out_rows)

    # Keep original columns + new ones
    keep_cols = list(df_demand.columns) + ['cases_assigned', 'cbm_assigned', 'container']
    result = result[keep_cols]

    # Nice ordering (container after group keys; include product_part_number if present)
    order_front = list(group_cols) + ['container']
    if 'product_part_number' in result.columns and 'product_part_number' not in order_front:
        order_front += ['product_part_number']
    other_cols = [c for c in result.columns if c not in order_front]
    result = result[order_front + other_cols]

    return result.reset_index(drop=True)

def build_cbm_capacity_map(
    df_demand: pd.DataFrame,
    cbm_max_df: pd.DataFrame,
    *,
    vendor_col: str = "CHW_PRIMARY_SUPPLIER_NUMBER",
    cbm_vendor_col: str = "vendor_number",
    cbm_capacity_col: str = "CBM Max",
    default_capacity: float = 66.0) -> Dict[Any, float]:
    """
    Create a {vendor -> CBM_Max} map using vendors present in df_demand.
    - cbm_max_df has vendor_number and 'CBM Max' columns (as given).
    - Any vendor missing in cbm_max_df gets default_capacity (66 by default).
    """
    if df_demand.empty:
        return {}

    # Normalize vendor ids to string to avoid type mismatches (e.g., 1001 vs "1001")
    demand_vendors = (
        df_demand[vendor_col]
        .astype("string")
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    # Clean CBM_Max table
    cap_tbl = cbm_max_df[[cbm_vendor_col, cbm_capacity_col]].copy()
    cap_tbl[cbm_vendor_col] = cap_tbl[cbm_vendor_col].astype("string").str.strip()
    # coerce to numeric, drop null/invalid capacities
    cap_tbl[cbm_capacity_col] = pd.to_numeric(cap_tbl[cbm_capacity_col], errors="coerce")

    # Keep last occurrence if duplicates
    cap_tbl = cap_tbl.dropna(subset=[cbm_vendor_col, cbm_capacity_col]).drop_duplicates(
        subset=[cbm_vendor_col], keep="last"
    )

    # Build raw map from CBM_Max
    raw_map: Dict[str, float] = dict(zip(cap_tbl[cbm_vendor_col], cap_tbl[cbm_capacity_col]))

    # Final map for vendors found in demand; default if missing/invalid
    out: Dict[Any, float] = {}
    for v in demand_vendors:
        cap = raw_map.get(v, default_capacity)
        try:
            cap = float(cap)
            if cap <= 0:
                cap = default_capacity
        except Exception:
            cap = default_capacity
        out[v] = cap

    return out


def process_DQ_workflow():
    df_demand = sql_lite_store.load_table("demand_data")
    df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
    df_vendor_cbm = sql_lite_store.load_table("Vendor_CBM")

    intermediate_demand, DQ_skus_w_no_splits, DQ_skus_w_no_record_in_plm, DQ_excluded_skus_w_incomplete_data = process_and_locate_missing_records(df_demand, df_kepplerSplits, df_vendor_cbm)
    ok, count = sql_lite_store.save_table(intermediate_demand, "intermediate_demand")
    ok, count = sql_lite_store.save_table(DQ_skus_w_no_record_in_plm, "DQ_skus_w_no_record_in_plm")
    ok, count = sql_lite_store.save_table(DQ_excluded_skus_w_incomplete_data, "DQ_excluded_skus_w_incomplete_data")
    ok, count = sql_lite_store.save_table(DQ_skus_w_no_splits, "DQ_skus_w_no_splits")


def process_containerPlan_workflow():
    intermediate_demand = sql_lite_store.load_table("intermediate_demand")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")
    df_cbm_capacity_map = build_cbm_capacity_map(intermediate_demand, df_CBM_Max)
    output_df_containerPlan = build_containers_v2(intermediate_demand, df_cbm_capacity_map, group_cols=('CHW_PRIMARY_SUPPLIER_NUMBER','DEST'), vendor_col='CHW_PRIMARY_SUPPLIER_NUMBER')
        
    ok, count = sql_lite_store.save_table(output_df_containerPlan, "output_df_containerPlan")
    to_po_layout(output_df_containerPlan)


TARGET_COLUMNS = [
    "Group", "PO Number", "Supplier", "Location",
    "Expected pickup date", "Expected Delivery Date",
    "Buyer Code", "FOB code", "Line_number", "Item_ID",
    "Supplier UOM", "Quantity", "Shipping Agent Code",
    "Shipping Agent Name", "Tracking Number",
    "Closed for Finance Indicator", "Supplier Acknowledgement",
    " Import Type", "EDI 850 or Email Exempt", "Cancellation Reason"
]

def to_po_layout(df: pd.DataFrame) -> pd.DataFrame:
    # Safe numeric conversions
    case_pack = pd.to_numeric(df.get("CHW_MASTER_CASE_PACK", 0), errors="coerce").fillna(0)
    cases_assigned = pd.to_numeric(df.get("cases_assigned", 0), errors="coerce").fillna(0)

    # Quantity = case pack * cases assigned (rounded to int)
    qty = (case_pack * cases_assigned).round().astype(int)

    location = df["DEST"].astype(str) 


    out = pd.DataFrame({
        "Group": df["container"],                                    # container -> Group
        "PO Number": "",                                             # blank
        "Supplier": df["CHW_PRIMARY_SUPPLIER_NUMBER"],               # supplier number
        "Location": location,                                        # DEST
        "Expected pickup date": "",                                  # blank
        "Expected Delivery Date": "",                                # blank
        "Buyer Code": "",                                            # blank
        "FOB code": "",                                              # blank
        "Line_number": "",                                 # simple 1..N
        "Item_ID": df["CHW_SKU_NUMBER"],                             # SKU -> Item_ID
        "Supplier UOM": "EA",                                        # constant
        "Quantity": qty,                                             # case pack * cases_assigned
        "Shipping Agent Code": "",                                   # blank
        "Shipping Agent Name": "",                                   # blank
        "Tracking Number": "",                                       # blank
        "Closed for Finance Indicator": "",                          # blank
        "Supplier Acknowledgement": "",                              # blank
        " Import Type": "Lines",                                          # note leading space to match your header
        "EDI 850 or Email Exempt": "",                               # blank
        "Cancellation Reason": "Lines",                              # constant per your sample
    })

    # Ensure exact column order
    out = out[TARGET_COLUMNS]
    ok, count = sql_lite_store.save_table(out, "output_df_po_layout")

    return out



if __name__ == "__main__":

    process_DQ_workflow()
    process_containerPlan_workflow()
    



    # df_demand = sql_lite_store.load_table("demand_data")
    # df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
    # df_vendor_cbm = sql_lite_store.load_table("Vendor_CBM")
    # df_CBM_Max = sql_lite_store.load_table("CBM_Max")

    # df_demand_skinny, skus_w_no_splits, skus_w_no_record_in_plm, excluded_skus_w_incomplete_data = process_and_locate_missing_records(df_demand, df_kepplerSplits, df_vendor_cbm)
    # df_cbm_capacity_map = build_cbm_capacity_map(df_demand_skinny, df_CBM_Max)

    # containerPlan = build_containers(df_demand_skinny, df_cbm_capacity_map, group_cols=('CHW_PRIMARY_SUPPLIER_NUMBER','DEST'), vendor_col='CHW_PRIMARY_SUPPLIER_NUMBER')
    # df_containerPlan= pd.DataFrame(containerPlan)
    # df_containerPlan.to_csv('containerPlan.csv', index=False)

    # print(df_containerPlan.head())


