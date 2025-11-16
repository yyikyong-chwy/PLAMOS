
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class containerAssignmentState(BaseModel):
    container_id: int
    vendor_Code: str
    DEST: Literal["MDT1", "TLA1", "TNY1"]
    product_part_number: str
    cases_assigned: int
    cbm_per_case: float
    cbm_assigned: float