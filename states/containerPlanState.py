
from __future__ import annotations
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from states.containerState import containerState
from states.containerAssignmentState import containerAssignmentState

class containerPlanState(BaseModel):
    # One plan per (vendor, DEST) group with many containers and assignments
    vendor_Code: str
    DEST: Literal["MDT1", "TLA1", "TNY1"]
    containers: List[containerState] = []
    assignments: List[containerAssignmentState] = []