# state_loader.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional, Literal
import pandas as pd
import numpy as np
import math
from typing import Any

# --- Your state models (robust imports for flat or package layouts) ---
try:
    from ChewySkuState import ChewySkuState  # :contentReference[oaicite:0]{index=0}
except ImportError:
    from states.ChewySkuState import ChewySkuState  # alt layout

try:
    from vendorState import vendorState  # :contentReference[oaicite:1]{index=1}
except ImportError:
    from states.vendorState import vendorState

try:
    from containerPlanState import containerPlanState  # :contentReference[oaicite:2]{index=2}
    from containerState import containerState          # :contentReference[oaicite:3]{index=3}
    from containerAssignmentState import containerAssignmentState  # :contentReference[oaicite:4]{index=4}
except ImportError:
    from states.containerPlanState import containerPlanState
    from states.containerState import containerState
    from states.containerAssignmentState import containerAssignmentState



# ---------- Loader ----------
def _to_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _to_str(x: Any, default: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return default
    return str(x)


# -------------------------
# 1) DataFrame -> State(s)
# -------------------------

DEST = Literal["MDT1", "TLA1", "TNY1"]
def df_to_chewy_sku_states(
    df_sku_data: pd.DataFrame,
    df_splits: Optional[pd.DataFrame] = None,) -> List[ChewySkuState]:
    """
    Convert a SKU-level dataframe (already in the target schema) into List[ChewySkuState].

    Expected df_sku_data.columns:
        ['parent_product_part_number','product_part_number','product_name',
         'vendor_Code','vendor_name','vendor_purchaser_code','MOQ','MCP',
         'case_pk_CBM','planned_demand','vendor_earliest_ETD','MC1','MC2',
         'BRAND','SKU','PUBBED','OH','T90_DAILY_AVG','F90_DAILY_AVG',
         'AVG_LT','ost_ord','Next_Delivery','T90_DOS_OH','F90_DOS_OH',
         'F90_DOS_OO','T90_BELOW','F90_BELOW','baseConsumption',
         'bufferConsumption','baseDemand','bufferDemand','excess_demand']

    Optionally, df_splits can contain Kepler split fractions to build `demand_by_dest`.
      Expected columns (case-insensitive): ITEM_ID, MDT1_FRAC, TLA1_FRAC, TNY1_FRAC
      ITEM_ID should match product_part_number.
    """

    dfx = df_sku_data.copy()
    dfx.columns = [c.strip() for c in dfx.columns]
    if "product_part_number" in dfx.columns:
        dfx["product_part_number"] = dfx["product_part_number"].astype(str)

    # --- Build demand_by_dest from df_splits, if present ---
    demand_fractions: Dict[str, Dict[str, float]] = {}
    if df_splits is not None and not df_splits.empty:
        sp = df_splits.copy()
        sp.columns = [c.upper().strip() for c in sp.columns]
        if "ITEM_ID" in sp.columns:
            sp["ITEM_ID"] = sp["ITEM_ID"].astype(str)
            frac_map = {"MDT1_FRAC": "MDT1", "TLA1_FRAC": "TLA1", "TNY1_FRAC": "TNY1"}
            for _, r in sp.iterrows():
                sku = r["ITEM_ID"]
                demand_fractions[sku] = {
                    dest: (float(r.get(col)) if pd.notna(r.get(col)) else 0.0)
                    for col, dest in frac_map.items()
                }

    # --- Row -> ChewySkuState ---
    out: List[ChewySkuState] = []
    for _, r in dfx.iterrows():
        row = r.to_dict()

        # attach demand_by_dest if we have fractions + planned_demand
        demand_by_dest = None
        sku_key = str(row.get("product_part_number", ""))
        plan = row.get("planned_demand")
        if sku_key and plan is not None and sku_key in demand_fractions:
            fracs = demand_fractions[sku_key]
            demand_by_dest = {d: float(plan) * float(fr) for d, fr in fracs.items()}

        cs = ChewySkuState(
            parent_product_part_number=str(row.get("parent_product_part_number")),
            product_part_number=row.get("product_part_number"),
            product_name=_to_str(row.get("product_name")),
            vendor_Code=row.get("vendor_Code"),
            vendor_name=_to_str(row.get("vendor_name")),
            vendor_purchaser_code=_to_str(row.get("vendor_purchaser_code")),
            MOQ=_to_int(row.get("MOQ")),
            MCP=_to_int(row.get("MCP")),
            case_pk_CBM=_to_float(row.get("case_pk_CBM")),
            planned_demand=_to_float(row.get("planned_demand")),
            vendor_earliest_ETD=_to_str(row.get("vendor_earliest_ETD")),
            MC1=_to_str(row.get("MC1")),
            MC2=_to_str(row.get("MC2")),
            BRAND=_to_str(row.get("Brand")),
            PUBBED=_to_str(row.get("PUBBED")) if row.get("PUBBED") is not None else None,
            OH=_to_int(row.get("OH")),
            T90_DAILY_AVG=_to_float(row.get("T90_DAILY_AVG")),
            F90_DAILY_AVG=_to_float(row.get("F90_DAILY_AVG")),
            AVG_LT=_to_int(row.get("AVG_LT")),
            ost_ord=_to_int(row.get("ost_ord")),
            Next_Delivery=_to_str(row.get("Next_Delivery")) if row.get("Next_Delivery") is not None else None,
            T90_DOS_OH=_to_float(row.get("T90_DOS_OH")),
            F90_DOS_OH=_to_float(row.get("F90_DOS_OH")),
            F90_DOS_OO=_to_float(row.get("F90_DOS_OO")),
            T90_BELOW=_to_float(row.get("T90_BELOW")),
            F90_BELOW=_to_float(row.get("F90_BELOW")),
            baseConsumption=_to_float(row.get("baseConsumption")),
            bufferConsumption=_to_float(row.get("bufferConsumption")),
            baseDemand=_to_float(row.get("baseDemand")),
            bufferDemand=_to_float(row.get("bufferDemand")),
            excess_demand=_to_float(row.get("excess_demand")),
            demand_by_dest=demand_by_dest,
        )
        out.append(cs)

    return out


def df_to_vendor_states(df_sku_data: pd.DataFrame, df_vendor_cbm: pd.DataFrame, chewySkuStates: List[ChewySkuState]) -> List[vendorState]:
    out: List[vendorState] = []

    unique_vendors = df_sku_data[["vendor_Code", "vendor_name"]].drop_duplicates()
    for _, r in unique_vendors.iterrows():
        vs = vendorState(
            vendor_Code=str(r.get("vendor_Code")),
            vendor_name=_to_str(r.get("vendor_name")),
            CBM_Max = _to_float(
                df_vendor_cbm.loc[df_vendor_cbm["vendor_number"] == str(r.get("vendor_Code")), "CBM Max"].values[0]
            ) if not df_vendor_cbm.loc[df_vendor_cbm["vendor_number"] == str(r.get("vendor_Code")), "CBM Max"].empty else 66.0,

            Demand_skus=[sku for sku in chewySkuStates if sku.vendor_Code == str(r.get("vendor_Code"))],
        )
        out.append(vs)

    return out


    for _, r in df_vendor.iterrows():
        vs = vendorState(
            vendor_Code=str(r.get("vendor_Code")),
            vendor_name=_to_str(r.get("vendor_name")),
            CBM_Max=_to_float(r.get("CBM_Max")),
        )
        out.append(vs)
    return out
