from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

from states.ChewySkuState import ChewySkuState
from states.containerPlanState import ContainerPlanState

"""
each vendor would have a list of container plans
each container plan contains the following:
1. sku to container assignments
2. metrics to evaluate goodness of that particular plan
"""
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
    #this is just serving as lookup table for VF level data
    ChewySku_info: List[ChewySkuState] = Field(default_factory=list)    

    #this is the container plans for this vendor, maybe one, or more
    container_plans: List[ContainerPlanState] = Field(default_factory=list)

    def numberofPlans(self) -> int:
        return len(self.container_plans)

    def get_ith_df_container_plan(self, i: int) -> pd.DataFrame:
        return self.container_plans[i].to_df()

    def insert_container_plan(self, container_plan: ContainerPlanState) -> None:
        self.container_plans.append(container_plan)



