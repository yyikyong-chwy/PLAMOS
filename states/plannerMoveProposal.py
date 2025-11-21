from __future__ import annotations
from typing import Any, Dict, Optional, Literal
import json, re
import pandas as pd
from pydantic import BaseModel, Field

ActionType = Literal["reduce", "consolidate", "pad"]

class Reduce(BaseModel):
    cbm_goal: float

class Consolidate(BaseModel):
    from_dest: str
    from_container: int
    to_dest: str
    to_container: int
    cbm_move: float

class Pad(BaseModel):
    dest: str
    container: int
    cbm_goal: float

class OneMoveProposal(BaseModel):
    action: ActionType
    rationale: str = Field(default="", max_length=1500)
    reduce: Optional[Reduce] = None
    consolidate: Optional[Consolidate] = None
    pad: Optional[Pad] = None
