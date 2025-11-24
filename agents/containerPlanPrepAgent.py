from __future__ import annotations
from typing import Optional
from copy import deepcopy

from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState, PlanStrategy
from states.ContainerPlanMetrics import ContainerPlanMetrics
from states.ContainerRow import ContainerPlanRow
from states.planStrategy import next_strategy

def containerPlanPrepAgent(
    vendor: vendorState,
    *,
    copy_from_index: int = -1,             # which existing plan to copy (default: latest)
    strip_assignments: bool = False,        # if True, clear container/cases/cbm so downstream nodes can recompute
    reset_metrics: bool = False             # if True, start with fresh metrics
) -> vendorState:
    """
    Duplicate an existing ContainerPlanState and append it to vendor.container_plans.
    - If strip_assignments=True, sets (container, cases_assigned, cbm_assigned) to None on every row.
    - If reset_metrics=True, replaces metrics with a fresh ContainerPlanMetrics().

    Returns the mutated vendor state.
    """
    if not vendor.container_plans:
        # Nothing to copy; just return as-is.
        return vendor

    copy_from_index = 0 #always start from the first plan
    latest_plan: ContainerPlanState = vendor.container_plans[-1]
    src_plan: ContainerPlanState = vendor.container_plans[copy_from_index]    

    # Deep-copy rows and (optionally) wipe assignments so the next agent can re-compute
    new_rows: list[ContainerPlanRow] = []
    for r in src_plan.container_plan_rows:
        data = r.model_dump()
        if strip_assignments:
            data.update({
                "container": None,
                "cases_assigned": None,
                "cbm_assigned": None,
            })
        new_rows.append(ContainerPlanRow.model_validate(data))

    new_plan_strategy = next_strategy(getattr(latest_plan, "strategy", PlanStrategy.BASE_PLAN))

    # Build the new plan
    new_plan = ContainerPlanState(
        vendor_Code=vendor.vendor_Code,
        vendor_name=vendor.vendor_name,
        strategy=new_plan_strategy,
        container_plan_rows=new_rows,
        metrics=ContainerPlanMetrics() if reset_metrics else deepcopy(src_plan.metrics),
    )

    vendor.container_plans.append(new_plan)
    return vendor
