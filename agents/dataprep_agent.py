# dataprep_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
from datetime import datetime
import os
import logging
import pandas as pd
import numpy as np

# If you use the state model from the earlier message, import it:
# from state_model import State, upsert_sku_info, bulk_upsert_skus, log
# For a standalone wireframe, we retype minimal aliases:
State = Dict[str, Any]

logger = logging.getLogger("ACOS.DataPrep")
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class DataPrepConfig:
    # Optional file paths (if dataframes are not already in state["inputs"])
    demand_path: Optional[str] = None
    inventory_path: Optional[str] = None
    vendor_caps_path: Optional[str] = None
    cbm_map_path: Optional[str] = None
    moq_path: Optional[str] = None
    runrate_path: Optional[str] = None
    cost_path: Optional[str] = None

    # Column names
    sku_col: str = "CHW_SKU_NUMBER"
    vendor_col: str = "CHW_PRIMARY_SUPPLIER_NUMBER"
    dest_col: str = "DEST"
    demand_col: str = "Demand"
    case_pack_col: str = "CHW_MASTER_CASE_PACK"
    cbm_per_case_col: str = "CHW_MASTER_CARTON_CBM"
    onhand_col: str = "OnHand"        # in inventory table

    # Destination split (if upstream demand isn’t split per DEST)
    dest_split: Dict[str, float] = None  # e.g., {"MDT1":0.33,"TLA1":0.34,"TNY1":0.33}

    # Constraints / thresholds (downstream nodes use these)
    soft_overflow_pct: float = 0.10
    max_weeks_of_supply: int = 8
    stockout_risk_limit: float = 0.0
    min_util_threshold: float = 0.85
    case_pack_enforced: bool = True
    respect_moq: bool = True

    def __post_init__(self):
        if self.dest_split is None:
            self.dest_split = {"MDT1": 0.33, "TLA1": 0.34, "TNY1": 0.33}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _read_csv_optional(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        logger.warning(f"[DataPrep] Path not found: {path}")
        return None
    return pd.read_csv(path)

def _ensure_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[DataPrep] {name} missing columns: {missing}")

def dq_filter_demand(df: pd.DataFrame, case_pack_col: str, cbm_col: str, demand_col: str
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows with valid case pack, CBM, and positive demand."""
    mask = (
        df[case_pack_col].notna() & (df[case_pack_col] > 0) &
        df[cbm_col].notna() & (df[cbm_col] > 0) &
        df[demand_col].notna() & (df[demand_col] > 0)
    )
    clean = df.loc[mask].copy().reset_index(drop=True)
    rejected = df.loc[~mask].copy().reset_index(drop=True)
    return clean, rejected

def split_by_dest_if_needed(df: pd.DataFrame, cfg: DataPrepConfig) -> pd.DataFrame:
    """If DEST already present, normalize dtype; else, fan out using cfg.dest_split."""
    if cfg.dest_col in df.columns:
        out = df.copy()
        out[cfg.dest_col] = out[cfg.dest_col].astype(str)
        return out.reset_index(drop=True)

    parts = []
    for dest, p in cfg.dest_split.items():
        chunk = df.copy()
        chunk[cfg.dest_col] = dest
        chunk[cfg.demand_col] = (chunk[cfg.demand_col] * float(p)).round(0)
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)

def build_capacity_map(df_vendor_caps: Optional[pd.DataFrame], vendor_col: str
                       ) -> Dict[str, Dict[float, bool]]:
    """
    Expect boolean columns like size_40, size_48, size_53, size_66, etc.
    Returns: {vendor: {40.0: True, 48.0: False, ...}}
    """
    if df_vendor_caps is None or df_vendor_caps.empty:
        return {}
    size_cols = [c for c in df_vendor_caps.columns if c.lower().startswith("size_")]
    out: Dict[str, Dict[float, bool]] = {}
    for _, r in df_vendor_caps.iterrows():
        vendor = str(r[vendor_col])
        sizes: Dict[float, bool] = {}
        for c in size_cols:
            try:
                sz = float(c.split("_", 1)[1])
                sizes[sz] = bool(r[c])
            except Exception:
                continue
        out[vendor] = sizes
    return out

def compute_dest_shortage_scores(df_inventory: Optional[pd.DataFrame],
                                 df_demand_long: pd.DataFrame,
                                 dest_col: str, sku_col: str, demand_col: str,
                                 onhand_col: str = "OnHand") -> Dict[str, float]:
    """
    Simple proxy: shortage score = (Σ max(Demand - OnHand, 0)) / Σ Demand by DEST.
    """
    if df_inventory is None or df_inventory.empty:
        by_dest = df_demand_long.groupby(dest_col)[demand_col].sum()
        total = by_dest.sum()
        return {d: (float(v) / total) if total > 0 else 0.0 for d, v in by_dest.items()}

    inv = df_inventory.copy()
    if onhand_col not in inv.columns:
        inv[onhand_col] = 0
    dem = df_demand_long.groupby([dest_col, sku_col], as_index=False)[demand_col].sum()
    merged = dem.merge(inv[[sku_col, dest_col, onhand_col]],
                       on=[sku_col, dest_col], how="left")
    merged[onhand_col] = merged[onhand_col].fillna(0)
    merged["Gap"] = (merged[demand_col] - merged[onhand_col]).clip(lower=0)
    g = merged.groupby(dest_col).agg({demand_col: "sum", "Gap": "sum"})
    g["score"] = np.where(g[demand_col] > 0, g["Gap"] / g[demand_col], 0.0)
    return g["score"].to_dict()


# --------------------------------------------------------------------------------------
# Main node
# --------------------------------------------------------------------------------------
def node_data_prep(state: State, cfg: Optional[DataPrepConfig] = None) -> State:
    """
    LLM-free DataPrep node:
      - Read inputs (from state["inputs"] or CSV paths)
      - Validate required columns
      - DQ filter and DEST split
      - Merge CBM/case pack if needed
      - Build vendor capacity map and shortage scores
      - Populate suppliers list and SKU registry
      - Set constraints for downstream nodes
    """
    cfg = cfg or DataPrepConfig()

    # Ensure base structures
    state.setdefault("meta", {})
    state.setdefault("inputs", {})
    state.setdefault("suppliers", {})
    state.setdefault("skus", {})
    state.setdefault("logs", [])

    # Meta
    state["meta"].setdefault("started_at", datetime.utcnow().isoformat())
    state["meta"].setdefault("version", "acos.v1")

    # Pull or read dataframes
    inp = state["inputs"]
    df_demand = inp.get("df_demand") or _read_csv_optional(cfg.demand_path)
    df_inventory = inp.get("df_inventory") or _read_csv_optional(cfg.inventory_path)
    df_vendor_caps = inp.get("df_vendor_caps") or _read_csv_optional(cfg.vendor_caps_path)
    df_cbm_map = inp.get("df_cbm_map") or _read_csv_optional(cfg.cbm_map_path)
    df_moq = inp.get("df_moq") or _read_csv_optional(cfg.moq_path)
    df_runrate = inp.get("df_runrate") or _read_csv_optional(cfg.runrate_path)
    df_cost = inp.get("df_cost") or _read_csv_optional(cfg.cost_path)

    if df_demand is None:
        raise ValueError("[DataPrep] df_demand is required (not provided in state or path).")

    # Ensure core columns
    _ensure_cols(df_demand, [cfg.sku_col, cfg.vendor_col, cfg.demand_col], "df_demand")

    # Attach CBM and case pack if missing
    if cfg.case_pack_col not in df_demand.columns or cfg.cbm_per_case_col not in df_demand.columns:
        if df_cbm_map is None:
            raise ValueError("[DataPrep] Demand missing CBM/CasePack and no df_cbm_map provided.")
        _ensure_cols(df_cbm_map, [cfg.sku_col, cfg.case_pack_col, cfg.cbm_per_case_col], "df_cbm_map")
        df_demand = df_demand.merge(
            df_cbm_map[[cfg.sku_col, cfg.case_pack_col, cfg.cbm_per_case_col]],
            on=cfg.sku_col, how="left"
        )

    # DQ filtering
    clean, rejected = dq_filter_demand(df_demand, cfg.case_pack_col, cfg.cbm_per_case_col, cfg.demand_col)
    state["inputs"]["dq_rejected_demand"] = rejected  # keep for DQ tab if desired

    # Split by DEST if needed
    df_long = split_by_dest_if_needed(clean, cfg)

    # Build vendor capacity map
    capacity_map = build_capacity_map(df_vendor_caps, cfg.vendor_col)

    # Shortage scores (by DEST)
    dest_shortage_scores = compute_dest_shortage_scores(
        df_inventory=df_inventory,
        df_demand_long=df_long,
        dest_col=cfg.dest_col,
        sku_col=cfg.sku_col,
        demand_col=cfg.demand_col,
        onhand_col=cfg.onhand_col
    )

    # Supplier list and bucket init
    suppliers = sorted(df_long[cfg.vendor_col].astype(str).unique().tolist())
    for sid in suppliers:
        state["suppliers"].setdefault(sid, {"variants": [], "base_plan_id": None, "best_plan_id": None})

    # Optional: register SKU attributes (run rate, MOQ, case pack, cbm)
    # (Safe if those tables are absent; we add what we can)
    if df_runrate is not None and (cfg.sku_col in df_runrate.columns):
        rr = df_runrate[[cfg.sku_col]].copy()
        # bring runrate if exists
        for col in ("run_rate", "RunRate", "RUN_RATE"):
            if col in df_runrate.columns:
                rr["run_rate"] = df_runrate[col]
                break
        # WoS optional
        for col in ("weeks_of_supply", "WoS", "WOS"):
            if col in df_runrate.columns:
                rr["weeks_of_supply"] = df_runrate[col]
                break
        # merge case pack & cbm from demand (clean)
        slim = clean[[cfg.sku_col, cfg.vendor_col, cfg.case_pack_col, cfg.cbm_per_case_col]].drop_duplicates(cfg.sku_col)
        rr = rr.merge(slim, on=cfg.sku_col, how="left")
        # MOQ optional
        if df_moq is not None and cfg.sku_col in df_moq.columns:
            for col in ("MOQ", "moq", "min_order_qty"):
                if col in df_moq.columns:
                    rr["moq"] = df_moq[col]
                    break
        # Upsert into state["skus"]
        for _, r in rr.iterrows():
            sku_key = str(r[cfg.sku_col])
            info = state["skus"].get(sku_key, {})
            info.update({
                "sku": sku_key,
                "supplier_id": str(r.get(cfg.vendor_col, "")),
                "run_rate": float(r.get("run_rate", np.nan)) if pd.notna(r.get("run_rate", np.nan)) else None,
                "weeks_of_supply": float(r.get("weeks_of_supply", np.nan)) if pd.notna(r.get("weeks_of_supply", np.nan)) else None,
                "moq": int(r.get("moq")) if pd.notna(r.get("moq", np.nan)) else None,
                "case_pack": int(r.get(cfg.case_pack_col)) if pd.notna(r.get(cfg.case_pack_col, np.nan)) else None,
                "cbm_per_case": float(r.get(cfg.cbm_per_case_col)) if pd.notna(r.get(cfg.cbm_per_case_col, np.nan)) else None,
            })
            state["skus"][sku_key] = info

    # Constraints bundle for downstream nodes
    constraints = {
        "soft_overflow_pct": cfg.soft_overflow_pct,
        "max_weeks_of_supply": cfg.max_weeks_of_supply,
        "stockout_risk_limit": cfg.stockout_risk_limit,
        "min_util_threshold": cfg.min_util_threshold,
        "case_pack_enforced": cfg.case_pack_enforced,
        "respect_moq": cfg.respect_moq,
    }

    # Persist prepared inputs back to state
    state["inputs"].update({
        "df_demand": df_demand,                 # post-join
        "df_inventory": df_inventory if df_inventory is not None else pd.DataFrame(),
        "df_vendor_caps": df_vendor_caps if df_vendor_caps is not None else pd.DataFrame(),
        "df_cbm_map": df_cbm_map if df_cbm_map is not None else pd.DataFrame(),
        "df_moq": df_moq if df_moq is not None else pd.DataFrame(),
        "df_runrate": df_runrate if df_runrate is not None else pd.DataFrame(),
        "df_cost": df_cost if df_cost is not None else pd.DataFrame(),

        "df_demand_long": df_long,
        "capacity_map": capacity_map,
        "dest_shortage_scores": dest_shortage_scores,
        "suppliers_list": suppliers,
        "constraints": constraints,
    })

    # Log summary
    summary = {
        "suppliers": len(suppliers),
        "rows_in": len(df_demand),
        "rows_clean": len(df_long),
        "rows_rejected": len(rejected),
        "dest_scores": dest_shortage_scores,
        "has_vendor_caps": bool(capacity_map),
    }
    state["logs"].append({
        "t": datetime.utcnow().isoformat(),
        "lvl": "INFO",
        "msg": "DataPrep complete",
        "summary": summary
    })
    logger.info(f"[DataPrep] Summary: {summary}")
    return state


# --------------------------------------------------------------------------------------
# Minimal LangGraph wiring (example)
# --------------------------------------------------------------------------------------
# from langgraph.graph import StateGraph, END
#
# def node_after_dataprep(state: State) -> State:
#     # Next node reads: state["inputs"]["df_demand_long"], ["capacity_map"], etc.
#     return state
#
# g = StateGraph(State)
# g.add_node("data_prep", lambda st: node_data_prep(st, DataPrepConfig(
#     demand_path="data/demand.csv",
#     cbm_map_path="data/cbm_map.csv",
#     vendor_caps_path="data/vendor_caps.csv",
#     inventory_path="data/inventory.csv",
# )))
# g.add_node("baseplan", node_after_dataprep)
# g.add_edge("data_prep", "baseplan")
# g.set_entry_point("data_prep")
# g.set_finish_point("baseplan")
# app = g.compile()
#
# # Run:
# out_state = app.invoke({})
