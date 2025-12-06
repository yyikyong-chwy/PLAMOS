from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

#technically this is not a state object, but it is used to store the data for the sku
#intend to get rid off later...
# ---------- Core entities ----------
class ChewySkuState(BaseModel):
    model_config = ConfigDict(revalidate_instances='never')
    
    product_part_number: str
    product_name: str
    vendor_Code: str
    vendor_name: Optional[str] = None
    MOQ: Optional[int] = None
    MCP: Optional[int] = Field(None, description="Master Case Pack")
    case_pk_CBM: Optional[float] = None
    planned_demand: Optional[float] = None
    MC1: Optional[str] = None
    MC2: Optional[str] = None
    MC3: Optional[str] = None
    BRAND: Optional[str] = None
    SKU: Optional[str] = None
    CHW_OTB: Optional[float] = None
    OH: Optional[int] = None
    T90_DAILY_AVG: Optional[float] = None
    F90_DAILY_AVG: Optional[float] = None
    F180_DAILY_AVG: Optional[float] = None
    AVG_LT: Optional[int] = None
    ost_ord: Optional[int] = None
    T90_DOS_OH: Optional[float] = None
    F90_DOS_OH: Optional[float] = None
    F90_DOS_OO: Optional[float] = None
    F180_DOS_OH: Optional[float] = None
    F180_DOS_OO: Optional[float] = None
    demand_within_LT: Optional[float] = None
    projected_OH_end_LT: Optional[float] = None
    avg_4wk_runrate: Optional[float] = None
    DOS_end_LT_days: Optional[float] = None
    projected_OH_end_LT_plus4w: Optional[float] = None
    DOS_end_LT_plus4w_days: Optional[float] = None
    DW_FCST: Optional[float] = None
    PRODUCT_MARGIN_PER_UNIT: Optional[float] = None
