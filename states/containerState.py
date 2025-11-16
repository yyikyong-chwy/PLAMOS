
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class containerState(BaseModel):
    container_id: int
    vendor_Code: str
    DEST: Literal["MDT1", "TLA1", "TNY1"]
    cbm_used: float = 0.0
    cbm_capacity: float = 66.0