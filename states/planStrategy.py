
from __future__ import annotations
from enum import Enum


class PlanStrategy(str, Enum):
    BASE_PLAN              = "basePlan" #split simply according to keppler, then assign containers ignoring its utilization %
    CONSOLIDATE_REDUCE     = "consolidate_reduce" #consolidate, then reduce. if last container is more than 70% full, then pad it up
    CONSOLIDATE_ONLY       = "consolidate_only"#consolidate only. if last container is more than 70% full, then pad it up
    CONSOLIDATE_AND_PAD    = "consolidate_and_pad" #consolidate, and if last container is not full, then pad it up
    PAD_ONLY               = "pad_only" #pad only. can select any container to pad up to 95% full.


#Explicit order (don’t rely on Enum definition order implicitly)
_STRATEGY_ORDER = [
    PlanStrategy.BASE_PLAN,
    PlanStrategy.CONSOLIDATE_REDUCE,
    PlanStrategy.CONSOLIDATE_ONLY,
    PlanStrategy.CONSOLIDATE_AND_PAD,
    PlanStrategy.PAD_ONLY,
]

_STRATEGY_ORDER = [
    PlanStrategy.BASE_PLAN,
    PlanStrategy.CONSOLIDATE_REDUCE,
]


def next_strategy(curr: PlanStrategy | str) -> PlanStrategy:
    """Return the next PlanStrategy (wraps around at the end)."""
    try:
        curr_enum = curr if isinstance(curr, PlanStrategy) else PlanStrategy(curr)
    except ValueError:
        # Unknown -> start the cycle
        return _STRATEGY_ORDER[0]
    idx = _STRATEGY_ORDER.index(curr_enum)
    return _STRATEGY_ORDER[(idx + 1) % len(_STRATEGY_ORDER)]