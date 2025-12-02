import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# project imports
from data.vendor_state_loader import load_all_vendor_states
from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerPlanMetrics import ContainerPlanMetrics


# ----------------- helpers ----------------- #

def _compute_plan_demand_aggregates(metrics: ContainerPlanMetrics) -> Dict[str, float]:
    """
    Using metrics.demand_met_by_sku (already computed in planEvalAgent),
    aggregate totals:
      - total_planned
      - total_assigned
      - unmet_total = sum(max(0, original - assigned))
      - exceed_total = sum(max(0, assigned - original))
      - met_total    = min(original, assigned) summed across SKUs
    Then compute percentages relative to total_planned (if > 0).
    """
    rows = metrics.demand_met_by_sku or []
    if not rows:
        return {
            "total_planned": 0.0,
            "total_assigned": 0.0,
            "unmet_total": 0.0,
            "exceed_total": 0.0,
            "met_total": 0.0,
            "pct_unmet": 0.0,
            "pct_exceed": 0.0,
            "pct_met": 0.0,
        }

    total_planned = 0.0
    total_assigned = 0.0
    unmet_total = 0.0
    exceed_total = 0.0
    met_total = 0.0

    for r in rows:
        orig = float(r.get("original_demand", 0.0) or 0.0)
        assigned = float(r.get("assigned_demand", 0.0) or 0.0)
        total_planned += orig
        total_assigned += assigned
        unmet_total += max(0.0, orig - assigned)
        exceed_total += max(0.0, assigned - orig)
        met_total += min(orig, assigned)

    if total_planned > 0:
        pct_unmet = 100.0 * unmet_total / total_planned
        pct_exceed = 100.0 * exceed_total / total_planned
        pct_met = 100.0 * met_total / total_planned
    else:
        pct_unmet = pct_exceed = pct_met = 0.0

    return {
        "total_planned": total_planned,
        "total_assigned": total_assigned,
        "unmet_total": unmet_total,
        "exceed_total": exceed_total,
        "met_total": met_total,
        "pct_unmet": pct_unmet,
        "pct_exceed": pct_exceed,
        "pct_met": pct_met,
    }


def _count_underutilized_containers(
    plan: ContainerPlanState,
    vendor: vendorState,
    threshold: float = 0.95,
) -> int:
    """
    Count containers whose utilization < threshold.
    Utilization = sum(cbm_assigned) / vendor.CBM_Max
    """
    df = plan.to_df()
    if df.empty:
        return 0

    df = df.copy()
    df["cbm_assigned"] = pd.to_numeric(df["cbm_assigned"], errors="coerce").fillna(0.0)

    cbm_max = float(getattr(vendor, "CBM_Max", 66.0) or 66.0)

    util_by_container = (
        df.groupby("container")["cbm_assigned"].sum() / cbm_max
    )

    return int((util_by_container < threshold).sum())


def _plan_summary_row(
    plan_idx: int,
    plan: ContainerPlanState,
    vendor: vendorState,
) -> Dict[str, Any]:
    """
    Build a single summary row (dict) for the comparison table across plans.
    """
    m = plan.metrics or ContainerPlanMetrics()
    agg = _compute_plan_demand_aggregates(m)
    underutilized_cnt = _count_underutilized_containers(plan, vendor, threshold=0.95)

    row = {
        "strategy": getattr(plan, "strategy", None),
        "#containers": int(getattr(m, "containers", 0) or 0),
        "avg_utilization_%": float(getattr(m, "avg_utilization", 0.0) or 0.0) * 100.0,
        "#containers_underutilized": underutilized_cnt,
        "total_planned_demand": agg["total_planned"],
        "total_assigned_demand": agg["total_assigned"],
        "demand_met_%": agg["pct_met"],
        "demand_unmet_total": agg["unmet_total"],
        "demand_unmet_%": agg["pct_unmet"],
        "demand_exceed_total": agg["exceed_total"],
        "demand_exceed_%": agg["pct_exceed"],
    }
    return row


def _build_plan_comparison_df(vendor: vendorState) -> pd.DataFrame:
    """
    For a given vendor, create a DataFrame summarizing all its plans side-by-side.
    """
    records: List[Dict[str, Any]] = []
    for i, plan in enumerate(vendor.container_plans):
        records.append(_plan_summary_row(i, plan, vendor))

    if not records:
        return pd.DataFrame(
            columns=[
                "strategy",
                "#containers",
                "avg_utilization_%",
                "#containers_underutilized",
                "total_planned_demand",
                "total_assigned_demand",
                "demand_met_%",
                "demand_unmet_total",
                "demand_unmet_%",
                "demand_exceed_total",
                "demand_exceed_%",
            ]
        )
    df = pd.DataFrame.from_records(records)
    return df


def _plan_detail_df(plan: ContainerPlanState) -> pd.DataFrame:
    """
    Build detailed table for a plan:
      container, SKU, demand (assigned eaches),
      cbm_assigned, cbm_cumulative (per container).
    """
    df = plan.to_df()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "container",
                "SKU",
                "Demand (eaches)",
                "CBM occupied",
                "CBM cumulative",
            ]
        )

    df = df.copy()

    # ensure numeric types
    for c in ("cases_assigned", "master_case_pack", "cbm_assigned", "case_pk_CBM"):
        if c in df.columns:
            if c in ("cases_assigned", "master_case_pack"):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # demand in eaches = cases_assigned * master_case_pack
    df["demand_assigned"] = (df["cases_assigned"] * df["master_case_pack"]).astype(int)

    # sort within container for consistent cumulative CBM
    sort_cols = ["container", "product_part_number"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # cumulative CBM per container
    df["cbm_cumulative"] = (
        df.groupby("container")["cbm_assigned"]
        .cumsum()
        .astype(float)
    )

    out = df[[
        "container",
        "product_part_number",
        "demand_assigned",
        "cbm_assigned",
        "cbm_cumulative",
    ]].copy()

    # rename columns for UI clarity
    out = out.rename(
        columns={
            "product_part_number": "SKU",
            "demand_assigned": "Demand (eaches)",
            "cbm_assigned": "CBM occupied",
            "cbm_cumulative": "CBM cumulative",
        }
    )

    # enforce integer display for demand
    out["Demand (eaches)"] = out["Demand (eaches)"].astype(int)

    return out


def _to_float(val) -> float:
    """Convert value to float, handling strings with commas."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        # Remove commas from formatted strings like "1,234"
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _style_exceed_unmet(row: pd.Series):
    """
    Highlight rows:
      - deep red if demand_unmet_total > 0
      - deep orange if demand_exceed_total > 0
    (unmet takes precedence if both > 0)
    """
    styles = [""] * len(row)

    unmet_total = _to_float(row.get("demand_unmet_total", 0))
    exceed_total = _to_float(row.get("demand_exceed_total", 0))

    if unmet_total > 0:
        # deep red
        style_str = "background-color: #b71c1c; color: white;"
        styles = [style_str] * len(row)
    elif exceed_total > 0:
        # deep orange
        style_str = "background-color: #e65100; color: white;"
        styles = [style_str] * len(row)

    return styles


def _short_strategy_name(strategy: str | None) -> str:
    """
    Take something like "planStrategy.BASE_PLAN" and return "BASE_PLAN".
    If None or empty, return "UNKNOWN".
    """
    if not strategy:
        return "UNKNOWN"
    return str(strategy).split(".")[-1]


# ----------------- Streamlit app ----------------- #

def main():
    st.set_page_config(
        page_title="Container Plan Comparison",
        layout="wide",
    )

    st.title("Container Plan Comparison Dashboard")

    # ---- Load all vendor states on launch ----
    with st.spinner("Loading vendor states..."):
        vendor_pairs = list(load_all_vendor_states(as_list=True))

    if not vendor_pairs:
        st.error("No vendor states found in data/vendor_plans.")
        return

    # Sidebar: pick vendor (code only)
    vendor_codes = [code for code, _ in vendor_pairs]
    code_to_state = {code: state for code, state in vendor_pairs}

    selected_vendor_code = st.sidebar.selectbox(
        "Select Vendor",
        options=vendor_codes,
        format_func=lambda x: x,  # code only
    )

    vendor = code_to_state[selected_vendor_code]

    st.subheader(f"Vendor: {vendor.vendor_Code}")

    if not vendor.container_plans:
        st.warning("This vendor has no container plans.")
        return

    # ---- Aggregated plan comparison ----
    st.markdown("### Aggregated Plan Metrics (for comparison)")

    comp_df = _build_plan_comparison_df(vendor)

    # ---- Aggregated plan comparison formatting ----
    if not comp_df.empty:
        comp_df_display = comp_df.copy()

        # --- Format % columns to 2 decimals ---
        pct_cols = [
            "avg_utilization_%",
            "demand_met_%",
            "demand_unmet_%",
            "demand_exceed_%",
        ]
        for c in pct_cols:
            if c in comp_df_display.columns:
                comp_df_display[c] = comp_df_display[c].apply(
                    lambda x: f"{x:.2f}"
                )

        # --- Format demand totals to integer (0 decimals) ---
        int_cols = [
            "total_planned_demand",
            "demand_unmet_total",
            "demand_exceed_total",
        ]
        for c in int_cols:
            if c in comp_df_display.columns:
                comp_df_display[c] = comp_df_display[c].apply(
                    lambda x: f"{x:.0f}"
                )

        # --- Apply row-level color highlighting (unmet/exceed) ---
        #styled = comp_df_display.style.apply(_style_exceed_unmet, axis=1)

        st.dataframe(
            comp_df_display,
            use_container_width=True,
        )
    else:
        st.info("No plan metrics available for this vendor yet.")


    st.markdown("---")

    # ---- Plan detail tabs ----
    st.markdown("### Plan Details")

    plan_labels = [
        _short_strategy_name(getattr(plan, "strategy", None))
        for plan in vendor.container_plans
    ]

    tabs = st.tabs(plan_labels)

    for plan, tab, label in zip(vendor.container_plans, tabs, plan_labels):
        with tab:
            st.markdown(f"**{label}**")

            detail_df = _plan_detail_df(plan)

            if detail_df.empty:
                st.info("No rows in this plan.")
            else:
                st.dataframe(
                    detail_df,
                    use_container_width=True,
                    height=500,
                )


if __name__ == "__main__":
    main()
