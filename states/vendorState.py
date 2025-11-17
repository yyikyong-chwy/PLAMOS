from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from states.ChewySkuState import ChewySkuState
from states.containerPlanState import containerPlanState


class vendorState(BaseModel):

    #Pydantic v2 by default revalidates model instances when they're passed to another model. 
    # Setting revalidate_instances='never' prevents this unnecessary revalidation and allows you to pass already-validated ChewySkuState objects directly.
    model_config = ConfigDict(
        revalidate_instances='never',
        arbitrary_types_allowed=True
    )

    vendor_Code: str
    vendor_name: Optional[str] = None
    CBM_Max: Optional[float] = Field(default=66.0, description="Default used if missing")
    Demand_skus: List[ChewySkuState] = Field(default_factory=list, description="SKU states for this vendor")

    #container_plans: List[containerPlanState] = Field(default_factory=list, description="Container plans for this vendor")


    def add_sku(self, sku: ChewySkuState) -> None:
        self.Demand_skus.append(sku)



