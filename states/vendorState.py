from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from states.ChewySkuState import ChewySkuState
from states.containerPlanState import containerPlanState


class vendorState(BaseModel):
    vendor_Code: str
    vendor_name: Optional[str] = None
    CBM_Max: Optional[float] = Field(default=66.0, description="Default used if missing")
    Demand_skus: List[ChewySkuState] = Field(default_factory=list, description="SKU states for this vendor")

    #container_plans: List[containerPlanState] = Field(default_factory=list, description="Container plans for this vendor")


    def add_sku(self, sku: ChewySkuState) -> None:
        self.skus.append(sku)



