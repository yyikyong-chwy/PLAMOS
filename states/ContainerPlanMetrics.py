
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Iterable, Union
from pydantic import BaseModel, Field
import pandas as pd

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

    total_cbm_used_by_container_dest: List[Dict[str, Union[str, int, float]]] = Field(default_factory=list)
    container_utilization_status_info: Optional[str] = None
    # Total EXCESS CBM attributed to each container (key = container id as str/int)
    # e.g. {"1": 8.4, "2": 0.0} or {1: 8.4, 2: 0.0}
    total_excess_in_cbm_by_container: Dict[Union[str, int], float] = Field(default_factory=dict)

    # Demand Met tracking by SKU (in eaches, summed across all locations)
    # Each dict contains: product_part_number, original_demand, assigned_demand, delta
    demand_met_by_sku: List[Dict[str, Union[str, float]]] = Field(default_factory=list)

    # final score (0–100)
    overall_score: float = 0.0

    notes: Optional[str] = None