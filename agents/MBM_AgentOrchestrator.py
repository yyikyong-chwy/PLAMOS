#manage this as a langgraph solution, to make it easier to manage the flow of the application

from __future__ import annotations
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from uuid import uuid4


