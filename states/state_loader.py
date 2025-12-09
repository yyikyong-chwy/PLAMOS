# state_loader.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional, Literal
import pandas as pd
import numpy as np
import math
from pydantic import ValidationError
from typing import Any

from states.ChewySkuState import ChewySkuState  # alt layout
from states.ContainerRow import ContainerPlanRow
from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.containerPlanState import PlanStrategy


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


def df_to_chewy_sku_states(
    df_sku_data: pd.DataFrame
) -> List[ChewySkuState]:
    """
    Loader for current df_sku_data schema:

    df_sku_data.columns:
      ['CHW_SKU_NUMBER', 'PLANNED_DEMAND', 'MC1_NAME', 'MC2_NAME', 'MC3_NAME',
       'BRAND', 'CHW_MOQ_LEVEL', 'CHW_PRIMARY_SUPPLIER_NAME',
       'CHW_PRIMARY_SUPPLIER_NUMBER', 'CHW_OTB', 'CHW_MASTER_CASE_PACK',
       'CHW_MASTER_CARTON_CBM', 'SKU', 'PRODUCT_NAME', 'OH', 'T90_DAILY_AVG',
       'F90_DAILY_AVG', 'F180_DAILY_AVG', 'AVG_LT', 'OO', 'T90_DOS_OH',
       'F90_DOS_OH', 'F90_DOS_OO', 'F180_DOS_OH', 'F180_DOS_OO',
       'PRODUCT_ABC_CODE', 'consumption_within_LT',
       'consumption_within_LT_plus4w', 'projected_OH_end_LT', 'runrate_at_LT',
       'DOS_end_LT_days', 'projected_OH_end_LT_plus4w', 'runrate_at_LT_plus4w',
       'DOS_end_LT_plus4w_days', 'PRODUCT_MARGIN_PER_UNIT']

    """
    dfx = df_sku_data.copy()
    dfx.columns = [c.strip() for c in dfx.columns]

    out: List[ChewySkuState] = []

    for _, r in dfx.iterrows():
        cs = ChewySkuState(
            # product identity
            product_part_number=_to_str(r.get("CHW_SKU_NUMBER")),
            product_name=_to_str(r.get("PRODUCT_NAME")),

            # vendor identity
            vendor_Code=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NUMBER")),
            vendor_name=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NAME")),

            # pack/constraints
            MOQ=_to_int(r.get("CHW_MOQ_LEVEL")),
            MCP=_to_int(r.get("CHW_MASTER_CASE_PACK")),
            case_pk_CBM=_to_float(r.get("CHW_MASTER_CARTON_CBM")),

            # demand fields
            planned_demand=_to_float(r.get("PLANNED_DEMAND")),
            MC1=_to_str(r.get("MC1_NAME")),
            MC2=_to_str(r.get("MC2_NAME")),
            MC3=_to_str(r.get("MC3_NAME")),
            BRAND=_to_str(r.get("BRAND")),
            SKU=_to_str(r.get("SKU")),
            CHW_OTB=_to_float(r.get("CHW_OTB")),

            # supply / rates
            OH=_to_int(r.get("OH")),
            T90_DAILY_AVG=_to_float(r.get("T90_DAILY_AVG")),
            F90_DAILY_AVG=_to_float(r.get("F90_DAILY_AVG")),
            F180_DAILY_AVG=_to_float(r.get("F180_DAILY_AVG")),
            AVG_LT=_to_int(r.get("AVG_LT")),
            ost_ord=_to_int(r.get("OO")),
            T90_DOS_OH=_to_float(r.get("T90_DOS_OH")),
            F90_DOS_OH=_to_float(r.get("F90_DOS_OH")),
            F90_DOS_OO=_to_float(r.get("F90_DOS_OO")),
            F180_DOS_OH=_to_float(r.get("F180_DOS_OH")),
            F180_DOS_OO=_to_float(r.get("F180_DOS_OO")),

            # classification
            product_abc_code=_to_str(r.get("PRODUCT_ABC_CODE")),

            # computed/projection fields
            demand_within_LT=_to_float(r.get("consumption_within_LT")),
            projected_OH_end_LT=_to_float(r.get("projected_OH_end_LT")),
            avg_4wk_runrate=_to_float(r.get("runrate_at_LT")),
            runrate_at_LT=_to_float(r.get("runrate_at_LT")),
            runrate_at_LT_plus4w=_to_float(r.get("runrate_at_LT_plus4w")),
            DOS_end_LT_days=_to_float(r.get("DOS_end_LT_days")),
            projected_OH_end_LT_plus4w=_to_float(r.get("projected_OH_end_LT_plus4w")),
            DOS_end_LT_plus4w_days=_to_float(r.get("DOS_end_LT_plus4w_days")),
            PRODUCT_MARGIN_PER_UNIT=_to_float(r.get("PRODUCT_MARGIN_PER_UNIT")),
        )
        out.append(cs)

    return out


def df_to_vendor_states(df_sku_data: pd.DataFrame, df_CBM_Max: pd.DataFrame, chewySkuStates: List[ChewySkuState], containerPlanRows: List[ContainerPlanRow]) -> List[vendorState]:
    out: List[vendorState] = []

    unique_vendors = df_sku_data[["CHW_PRIMARY_SUPPLIER_NUMBER", "CHW_PRIMARY_SUPPLIER_NAME"]].drop_duplicates()
    for _, r in unique_vendors.iterrows():
        vendor_code = str(r.get("CHW_PRIMARY_SUPPLIER_NUMBER"))
        cbm_max = _to_float(
            df_CBM_Max.loc[df_CBM_Max["vendor_number"] == vendor_code, "CBM Max"].values[0]
        ) if not df_CBM_Max.loc[df_CBM_Max["vendor_number"] == vendor_code, "CBM Max"].empty else 66.0
        demand_skus = [sku for sku in chewySkuStates if sku.vendor_Code == vendor_code]
        
        container_plan_state = ContainerPlanState.construct_state(vendor_Code=vendor_code, rows=containerPlanRows, vendor_name=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NAME")), strategy = PlanStrategy.BASE_PLAN,)

        # Use model_construct to bypass validation and avoid class identity issues during module reloads
        vs = vendorState.model_construct(
            vendor_Code=vendor_code,
            vendor_name=_to_str(r.get("CHW_PRIMARY_SUPPLIER_NAME")),
            CBM_Max=cbm_max,
            ChewySku_info=demand_skus,
            strategy=PlanStrategy.BASE_PLAN, #always start with base plan
            container_plans=[container_plan_state] #inserting a blank one for now, will add more later
        )
        out.append(vs)

    return out


def load_container_plan_rows(demand_by_Dest: pd.DataFrame) -> List[ContainerPlanRow]:
    """
    Map demand_by_Dest -> List[ContainerPlanRow].
    Required columns (must exist in the DataFrame):
      CHW_PRIMARY_SUPPLIER_NUMBER, CHW_PRIMARY_SUPPLIER_NAME, DEST, CHW_SKU_NUMBER,
      CHW_MASTER_CASE_PACK, CHW_MASTER_CARTON_CBM, baseDemandCasesNeed
    Optionally reads: container, cases_assigned (if present).
    """
    required = [
        "CHW_PRIMARY_SUPPLIER_NUMBER", "CHW_PRIMARY_SUPPLIER_NAME", "DEST",
        "CHW_SKU_NUMBER", "CHW_MASTER_CASE_PACK", "CHW_MASTER_CARTON_CBM",
        "PLANNED_DEMAND_cases_need"
    ]
    missing = [c for c in required if c not in demand_by_Dest.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    def _to_int(x):
        if x is None: return None
        try:
            if isinstance(x, float) and math.isnan(x): return None
            return int(x)
        except Exception:
            return None

    def _to_float(x):
        if x is None: return None
        try:
            return float(x)
        except Exception:
            return None

    rows: List[ContainerPlanRow] = []
    for rec in demand_by_Dest.to_dict(orient="records"):
        # Skip if any *critical* field is missing after coercion
        vendor_code = rec.get("CHW_PRIMARY_SUPPLIER_NUMBER")
        vendor_name = rec.get("CHW_PRIMARY_SUPPLIER_NAME")
        dest = rec.get("DEST")
        sku = rec.get("CHW_SKU_NUMBER")

        mcp = _to_int(rec.get("CHW_MASTER_CASE_PACK"))
        cbm = _to_float(rec.get("CHW_MASTER_CARTON_CBM"))
        need_cases = _to_int(rec.get("PLANNED_DEMAND_cases_need"))

        if not (vendor_code and vendor_name and dest and sku and mcp is not None and cbm is not None and need_cases is not None):
            # Row is incomplete for the model; skip it
            continue

        cases_assigned = rec.get("cases_assigned", None)
        cases_assigned = _to_int(cases_assigned) if cases_assigned is not None else None

        cbm_assigned = None
        if cases_assigned is not None and cbm is not None:
            cbm_assigned = cases_assigned * cbm

        try:
            item = ContainerPlanRow(
                vendor_Code=str(vendor_code),
                vendor_name=str(vendor_name),
                DEST=str(dest),
                container=_to_int(rec.get("container")),   # will be None (not in df)
                product_part_number=str(sku),
                master_case_pack=int(mcp),
                case_pk_CBM=float(cbm),
                cases_needed=int(need_cases),
                cases_assigned=cases_assigned,
                cbm_assigned=cbm_assigned,
            )
            rows.append(item)
        except ValidationError:
            # If any row still fails validation, skip it
            continue

    return rows
