
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Iterable
from pydantic import BaseModel, Field
import pandas as pd
from enum import Enum

from states.ContainerRow import ContainerPlanRow
from states.ContainerPlanMetrics import ContainerPlanMetrics
from states.plannerMoveProposal import OneMoveProposal
from states.planStrategy import PlanStrategy


# ---- State object holding the plan as rows, with DF helpers ----
class ContainerPlanState(BaseModel):
    vendor_Code: str
    vendor_name: Optional[str] = None
    
    strategy: PlanStrategy = Field(default=PlanStrategy.BASE_PLAN, description="Strategy for the container plan (default: basePlan)")
    moveProposal: Optional[OneMoveProposal] = None
    container_plan_rows: List[ContainerPlanRow] = Field(default_factory=list)
    metrics: ContainerPlanMetrics = Field(default_factory=ContainerPlanMetrics)
    plan_loop_counter: int = Field(default=1, description="Number of times the current plan has looped through the planner agent")

    # --- Fast conversion to DataFrame when processing inside nodes ---
    def to_df(self) -> pd.DataFrame:
        if not self.container_plan_rows:
            return pd.DataFrame(columns=list(ContainerPlanRow.model_json_schema()["properties"].keys()))

        records = [r.model_dump() for r in self.container_plan_rows]
        return pd.DataFrame.from_records(records)

    # --- Create state from a pandas DataFrame (e.g., your input sample) ---
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "ContainerPlanState":
        # Optional: enforce column order/rename here if needed
        rows: List[ContainerPlanRow] = []
        # Use itertuples for speed; let Pydantic validate/conform types
        for rec in df.to_dict("records"):
            rows.append(ContainerPlanRow(**rec))
        return cls(container_plan_rows=rows)

    # --- Append new assignments (e.g., from an assignment node) ---
    def extend_from_records(self, records: Iterable[dict]) -> None:
        for rec in records:
            self.container_plan_rows.append(ContainerPlanRow(**rec))
    
    @classmethod
    def construct_state(
        cls,
        vendor_Code: str,
        rows: List[ContainerPlanRow],
        vendor_name: Optional[str] = None,
        strategy: PlanStrategy = PlanStrategy.BASE_PLAN,
    ) -> "ContainerPlanState":
        """
        Create a ContainerPlanState for a single vendor by filtering the given rows.
        Only rows matching vendor_Code are included.
        """
        filtered = [r for r in rows if r.vendor_Code == vendor_Code]
        return cls(
            vendor_Code=vendor_Code,
            vendor_name=vendor_name,
            strategy=strategy,
            container_plan_rows=filtered,
            # metrics will use default_factory
            plan_loop_counter=1,
        )