from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---- Result schema the graph can consume ----
class CritiqueResult(BaseModel):
    action: Literal["proceed", "revise"]               # what the graph should do next
    reason: Optional[str] = Field(default="", max_length=1500)

