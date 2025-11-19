

from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class ContainerPlanRow(BaseModel):
    vendor_Code: str                           # CHW_PRIMARY_SUPPLIER_NUMBER
    vendor_name: str                           # CHW_PRIMARY_SUPPLIER_NAME
    DEST: str                                  # from DEST
    container: Optional[int] = None            # not in df yet
    product_part_number: str                   # CHW_SKU_NUMBER
    master_case_pack: int                      # CHW_MASTER_CASE_PACK
    case_pk_CBM: float                         # CHW_MASTER_CARTON_CBM

    cases_needed: int         # baseDemandCasesNeed

    # decision + simple derived
    cases_assigned: Optional[int]
    cbm_assigned: Optional[float]                  # = cases_assigned * case_pk_CBM (initially 0)
