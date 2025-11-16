from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class vendorState(BaseModel):
    vendor_Code: str
    CBM_Max: Optional[float] = Field(default=66.0, description="Default used if missing")
