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
    df_splits: Optional[pd.DataFrame] = None,
) -> List[ChewySkuState]:
    """
    Loader for current df_sku_data schema:

    df_sku_data.columns:
      ['CHW_SKU_NUMBER','Planned_Demand','MC1_NAME','MC2_NAME','MC3_NAME','BRAND',
       'CHW_MOQ_LEVEL','CHW_PRIMARY_SUPPLIER_NUMBER','CHW_OTB','CHW_MASTER_CASE_PACK',
       'CHW_MASTER_CARTON_CBM','SKU','PRODUCT_NAME','OH','T90_DAILY_AVG','F90_DAILY_AVG',
       'AVG_LT','OO','NEXT_DELIVERY','T90_DOS_OH','F90_DOS_OH','F90_DOS_OO','T90_BELOW',
       'F90_BELOW','ALERT','baseConsumption','bufferConsumption','baseDemand',
       'bufferDemand','excess_demand']

    df_splits (optional) expected columns (case-insensitive):
      ITEM_ID, MDT1_FRAC, TLA1_FRAC, TNY1_FRAC
      where ITEM_ID matches CHW_SKU_NUMBER
    """
    dfx = df_sku_data.copy()
    dfx.columns = [c.strip() for c in dfx.columns]

    # --- Build demand_by_dest fractions map keyed by CHW_SKU_NUMBER ---
    demand_fractions: Dict[str, Dict[str, float]] = {}
    if df_splits is not None and not df_splits.empty:
        sp = df_splits.copy()        
        sp = sp[sp["TOTAL_STAT_FCAST"] > 0]
        sp["CHW_SKU_NUMBER"] = sp["CHW_SKU_NUMBER"].astype(str).str.strip()
        sp.columns = [c.upper().strip() for c in sp.columns]
        frac_map = {"MDT1_FRAC": "MDT1", "TLA1_FRAC": "TLA1", "TNY1_FRAC": "TNY1"}
        for _, rr in sp.iterrows():
            sku = rr["CHW_SKU_NUMBER"]
            demand_fractions[sku] = {
                dest: (float(rr.get(col)) if pd.notna(rr.get(col)) else 0.0)
                for col, dest in frac_map.items()
            }

    out: List[ChewySkuState] = []

    for _, r in dfx.iterrows():
        # keys / aliases from your df
        sku_num = _to_str(r.get("CHW_SKU_NUMBER"))
        planned = _to_float(r.get("Planned_Demand"))

        # demand_by_dest if splits provided
        demand_by_dest = None
        if planned is not None and planned > 0:
            if sku_num and sku_num in demand_fractions:
                # Use provided splits
                fracs = demand_fractions[sku_num]
                demand_by_dest = {d: planned * float(fr) for d, fr in fracs.items()}
            else:
                # Default: 100% to TNY1
                demand_by_dest = {
                    "MDT1": 0.0,
                    "TLA1": 0.0,
                    "TNY1": planned
                }

        cs = ChewySkuState(
            # product identity
            #parent_product_part_number=_to_str(r.get("CHW_SKU_NUMBER")),                 # not present in your df
            product_part_number=_to_str(r.get("CHW_SKU_NUMBER")),                     # map CHW_SKU_NUMBER -> product_part_number
            product_name=_to_str(r.get("PRODUCT_NAME")), #this is the name of the product

            # vendor identity
            vendor_Code=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NUMBER")),
            vendor_name=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NAME")),                                # not present in df_sku_data
            vendor_purchaser_code=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NUMBER")),                      # not present

            # pack/constraints
            MOQ=_to_int(r.get("CHW_MOQ_LEVEL")),
            MCP=_to_int(r.get("CHW_MASTER_CASE_PACK")),
            case_pk_CBM=_to_float(r.get("CHW_MASTER_CARTON_CBM")),

            # demand fields
            planned_demand=_to_float(r.get("Planned_Demand")),
            vendor_earliest_ETD=None,                        # not present
            MC1=_to_str(r.get("MC1_NAME")),
            MC2=_to_str(r.get("MC2_NAME")),
            BRAND=_to_str(r.get("BRAND")),
            SKU=_to_str(r.get("CHW_SKU_NUMBER")),
            PUBBED=None,                                     # not present

            # supply / rates
            OH=_to_int(r.get("OH")),
            T90_DAILY_AVG=_to_float(r.get("T90_DAILY_AVG")),
            F90_DAILY_AVG=_to_float(r.get("F90_DAILY_AVG")),
            AVG_LT=_to_int(r.get("AVG_LT")),
            ost_ord=_to_int(r.get("OO")),
            Next_Delivery=_to_str(r.get("NEXT_DELIVERY")),
            T90_DOS_OH=_to_float(r.get("T90_DOS_OH")),
            F90_DOS_OH=_to_float(r.get("F90_DOS_OH")),
            F90_DOS_OO=_to_float(r.get("F90_DOS_OO")),
            T90_BELOW=_to_float(r.get("T90_BELOW")),
            F90_BELOW=_to_float(r.get("F90_BELOW")),

            # computed fields
            baseConsumption=_to_float(r.get("baseConsumption")),
            bufferConsumption=_to_float(r.get("bufferConsumption")),
            baseDemand=_to_float(r.get("baseDemand")),
            bufferDemand=_to_float(r.get("bufferDemand")),
            excess_demand=_to_float(r.get("excess_demand")),

            # per-destination demand (optional)
            demand_by_dest=demand_by_dest,
        )
        out.append(cs)

    return out


def df_to_vendor_states(df_sku_data: pd.DataFrame, df_CBM_Max: pd.DataFrame, chewySkuStates: List[ChewySkuState]) -> List[vendorState]:
    out: List[vendorState] = []

    unique_vendors = df_sku_data[["CHW_PRIMARY_SUPPLIER_NUMBER", "CHW_PRIMARY_SUPPLIER_NAME"]].drop_duplicates()
    for _, r in unique_vendors.iterrows():
        vendor_code = str(r.get("CHW_PRIMARY_SUPPLIER_NUMBER"))
        cbm_max = _to_float(
            df_CBM_Max.loc[df_CBM_Max["vendor_number"] == vendor_code, "CBM Max"].values[0]
        ) if not df_CBM_Max.loc[df_CBM_Max["vendor_number"] == vendor_code, "CBM Max"].empty else 66.0
        demand_skus = [sku for sku in chewySkuStates if sku.vendor_Code == vendor_code]
        
        # Use model_construct to bypass validation and avoid class identity issues during module reloads
        vs = vendorState.model_construct(
            vendor_Code=vendor_code,
            vendor_name=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NAME")),
            CBM_Max=cbm_max,
            Demand_skus=demand_skus,
        )
        out.append(vs)

    return out



