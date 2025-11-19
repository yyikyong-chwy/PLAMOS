from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

#technically this is not a state object, but it is used to store the data for the sku
#intend to get rid off later...
# ---------- Core entities ----------
class ChewySkuState(BaseModel):
    model_config = ConfigDict(revalidate_instances='never')
    
    parent_product_part_number: Optional[str] = None
    product_part_number: str
    product_name: str
    vendor_Code: str
    vendor_name: Optional[str] = None
    vendor_purchaser_code: Optional[str] = None
    MOQ: Optional[int] = None
    MCP: Optional[int] = Field(None, description="Master Case Pack")
    case_pk_CBM: Optional[float] = None
    planned_demand: Optional[float] = None
    vendor_earliest_ETD: Optional[str] = None
    MC1: Optional[str] = None
    MC2: Optional[str] = None
    BRAND: Optional[str] = None
    PUBBED: Optional[str] = None
    OH: Optional[int] = None
    T90_DAILY_AVG: Optional[float] = None
    F90_DAILY_AVG: Optional[float] = None
    AVG_LT: Optional[int] = None
    ost_ord: Optional[int] = None
    Next_Delivery: Optional[str] = None
    T90_DOS_OH: Optional[float] = None
    F90_DOS_OH: Optional[float] = None
    F90_DOS_OO: Optional[float] = None
    T90_BELOW: Optional[float] = None
    F90_BELOW: Optional[float] = None
    baseConsumption: Optional[float] = None
    bufferConsumption: Optional[float] = None
    baseDemand: Optional[float] = None
    bufferDemand: Optional[float] = None
    excess_demand: Optional[float] = None

