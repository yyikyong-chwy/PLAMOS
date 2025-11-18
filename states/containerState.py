
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, field_validator, model_validator

DEST = Literal["MDT1", "TLA1", "TNY1"]

class ContainerState(BaseModel):
    container_id: int
    vendor_Code: str
    DEST: DEST
    cbm_capacity: float = 66.0
    cbm_used: float = 0.0
    container_type: Optional[str] = None
    etd: Optional[str] = None
    eta: Optional[str] = None
    locked: bool = False

