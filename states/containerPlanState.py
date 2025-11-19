
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Iterable
from pydantic import BaseModel, Field
import pandas as pd

from states.ContainerRow import ContainerPlanRow


class ContainerPlanMetrics(BaseModel):
    containers: int = 0
    total_cbm_used: float = 0.0
    total_cbm_capacity: float = 0.0
    avg_utilization: float = 0.0
    weighted_utilization: float = 0.0
    demand_shortfall: float = 0.0
    sku_splits_off_mcp: int = 0
    moq_violations: int = 0
    low_util_count: int = 0               # # of containers < 90% utilization
    low_util_threshold: float = 0.90

    # SKU-alignment errors (weighted MAPE-like, in [0, 1])
    ape_vs_planned: float = 0.0
    ape_vs_base: float = 0.0
    ape_vs_excess: float = 0.0

    # final score (0–100)
    overall_score: float = 0.0

    notes: Optional[str] = None



PlanType = Literal["base", "alternate"]

# ---- State object holding the plan as rows, with DF helpers ----
class ContainerPlanState(BaseModel):
    vendor_Code: str
    vendor_name: Optional[str] = None
    
    plan_type: PlanType = Field(default="base", description="Type of container plan (default: base)")
    container_plan_rows: List[ContainerPlanRow] = Field(default_factory=list)
    metrics: ContainerPlanMetrics = Field(default_factory=ContainerPlanMetrics)

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
        plan_type: PlanType = "base",
    ) -> "ContainerPlanState":
        """
        Create a ContainerPlanState for a single vendor by filtering the given rows.
        Only rows matching vendor_Code are included.
        """
        filtered = [r for r in rows if r.vendor_Code == vendor_Code]
        return cls(
            vendor_Code=vendor_Code,
            vendor_name=vendor_name,
            plan_type=plan_type,
            container_plan_rows=filtered,
            # metrics will use default_factory
        )