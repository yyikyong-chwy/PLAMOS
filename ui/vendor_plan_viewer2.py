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
from states.ChewySkuState import ChewySkuState


# ----------------- helpers ----------------- #

def _compute_plan_demand_aggregates(metrics: ContainerPlanMetrics) -> Dict[str, float]:
    """
    Using metrics.demand_met_by_sku (already computed in planEvalAgent),
    aggregate totals and percentages.
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
      container, DEST, SKU, demand (assigned eaches),
      cbm_assigned, cbm_cumulative (per container).
    """
    df = plan.to_df()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "container",
                "DEST",
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

    # DEST column already exists in the data from ContainerPlanRow

    out = df[[
        "container",
        "DEST",
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


def _style_agg_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Style function for aggregated metrics:
      - If demand_unmet_total > 0: color demand_unmet_total and demand_unmet_% dark red.
      - If demand_exceed_total > 0: color demand_exceed_total and demand_exceed_% orange.
    """
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for idx, row in df.iterrows():
        unmet_total = _to_float(row.get("demand_unmet_total", 0))
        exceed_total = _to_float(row.get("demand_exceed_total", 0))

        if unmet_total > 0:
            styles.loc[idx, "demand_unmet_total"] = "background-color: #b71c1c; color: white;"
            styles.loc[idx, "demand_unmet_%"] = "background-color: #b71c1c; color: white;"

        if exceed_total > 0:
            styles.loc[idx, "demand_exceed_total"] = "background-color: #e65100; color: white;"
            styles.loc[idx, "demand_exceed_%"] = "background-color: #e65100; color: white;"

    return styles


def _short_strategy_name(strategy: str | None) -> str:
    """
    Take something like "planStrategy.BASE_PLAN" and return "BASE_PLAN".
    If None or empty, return "UNKNOWN".
    """
    if not strategy:
        return "UNKNOWN"
    return str(strategy).split(".")[-1]


def _build_sku_info_map(vendor: vendorState) -> Dict[str, ChewySkuState]:
    """
    Introspect vendor to find any attribute that is a List[ChewySkuState]
    and build a map: product_part_number -> ChewySkuState.
    """
    sku_map: Dict[str, ChewySkuState] = {}

    for attr_name in dir(vendor):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(vendor, attr_name)
        except AttributeError:
            continue

        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, ChewySkuState):
                for cs in val:
                    sku_map[str(cs.product_part_number)] = cs

    return sku_map


def _sku_tooltip_html(cs: ChewySkuState) -> str:
    """
    Build a small HTML table to show as a tooltip for a SKU cell.
    """
    fields = [
        ("product_part_number", cs.product_part_number),
        ("product_name", cs.product_name),
        ("MOQ", cs.MOQ),
        ("MCP", cs.MCP),
        ("case_pk_CBM", cs.case_pk_CBM),
        ("planned_demand", cs.planned_demand),
        ("vendor_earliest_ETD", cs.vendor_earliest_ETD),
        ("MC1", cs.MC1),
        ("MC2", cs.MC2),
        ("BRAND", cs.BRAND),
        ("OH", cs.OH),
        ("T90_DAILY_AVG", cs.T90_DAILY_AVG),
        ("F90_DAILY_AVG", cs.F90_DAILY_AVG),
        ("AVG_LT", cs.AVG_LT),
        ("ost_ord", cs.ost_ord),
        ("Next_Delivery", cs.Next_Delivery),
        ("T90_DOS_OH", cs.T90_DOS_OH),
        ("F90_DOS_OH", cs.F90_DOS_OH),
        ("F90_DOS_OO", cs.F90_DOS_OO),
        ("T90_BELOW", cs.T90_BELOW),
        ("F90_BELOW", cs.F90_BELOW),
        ("baseConsumption", cs.baseConsumption),
        ("bufferConsumption", cs.bufferConsumption),
        ("baseDemand", cs.baseDemand),
        ("bufferDemand", cs.bufferDemand),
        ("excess_demand", cs.excess_demand),
    ]

    rows_html = []
    for name, value in fields:
        v_str = "" if value is None else str(value)
        rows_html.append(
            "<tr>"
            f"<th style='padding:2px 6px;text-align:left;'>{name}</th>"
            f"<td style='padding:2px 6px;'>{v_str}</td>"
            "</tr>"
        )

    table_html = (
        "<table style='border-collapse:collapse;'>"
        f"{''.join(rows_html)}"
        "</table>"
    )
    return table_html


def _build_sku_details_df(
    skus: List[str], 
    sku_info_map: Dict[str, ChewySkuState], 
) -> pd.DataFrame:
    """
    Build a DataFrame with all SKU details for grid display.
    """
    records = []
    for sku in skus:
        if sku not in sku_info_map:
            continue
        cs = sku_info_map[sku]
        
        records.append({
            "SKU": cs.product_part_number,
            "Product Name": cs.product_name,
            "Planned Demand": cs.planned_demand,
            "MOQ": cs.MOQ,
            "MCP": cs.MCP,
            "CBM/Case": cs.case_pk_CBM,
            "OH": cs.OH,
            "T90 Daily Avg": cs.T90_DAILY_AVG,
            "F90 Daily Avg": cs.F90_DAILY_AVG,
            "Avg LT": cs.AVG_LT,
            "Base Demand": cs.baseDemand,
            "Buffer Demand": cs.bufferDemand,
            "Base Consumption": cs.baseConsumption,
            "Buffer Consumption": cs.bufferConsumption,
            "Excess Demand": cs.excess_demand,
            "T90 DOS OH": cs.T90_DOS_OH,
            "F90 DOS OH": cs.F90_DOS_OH,
            "F90 DOS OO": cs.F90_DOS_OO,
            "T90 Below": cs.T90_BELOW,
            "F90 Below": cs.F90_BELOW,
            "OST Ord": cs.ost_ord,
            "Next Delivery": cs.Next_Delivery,
            "Brand": cs.BRAND,
            "MC1": cs.MC1,
            "MC2": cs.MC2,
            "ETD": cs.vendor_earliest_ETD,
        })
    
    return pd.DataFrame(records)


def _build_po_lines_df(plan: ContainerPlanState, vendor: vendorState) -> pd.DataFrame:
    """
    Build the PO line-level grid from the actual plan data.

    Mapping:
      - Group → container
      - PO Number → blank
      - Location → DEST (actual destination like TNY1, TLA1, MDT1)
      - Item_ID → SKU (product_part_number)
      - Supplier UOM → "EA"
      - Quantity → Demand (eaches) = cases_assigned * master_case_pack
      - All other columns → blank
    """
    columns = [
        "Group",
        "PO Number",
        "Supplier",
        "Location",
        "Expected pickup date",
        "Expected Delivery Date",
        "Buyer Code",
        "FOB code",
        "Line_number",
        "Item_ID",
        "Supplier UOM",
        "Quantity",
        "Shipping Agent Code",
        "Shipping Agent Name",
        "Tracking Number",
        "Closed for Finance Indicator",
        "Supplier Acknowledgement",
        "Import Type",
        "EDI 850 or Email Exempt",
        "Cancellation Reason",
    ]

    # Get plan data
    plan_df = plan.to_df()

    if plan_df.empty:
        return pd.DataFrame(columns=columns)

    plan_df = plan_df.copy()

    # Ensure numeric types for demand calculation
    for c in ("cases_assigned", "master_case_pack"):
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce").fillna(0).astype(int)

    # Calculate demand in eaches
    plan_df["demand_eaches"] = (plan_df["cases_assigned"] * plan_df["master_case_pack"]).astype(int)

    # Sort by container and SKU for consistent ordering
    plan_df = plan_df.sort_values(["container", "product_part_number"]).reset_index(drop=True)

    # Build PO lines from plan data
    data = []
    for _, row in plan_df.iterrows():
        po_row = [
            row["container"],           # Group
            "",                          # PO Number (blank)
            "",                          # Supplier (blank)
            row["DEST"],                 # Location (actual destination like TNY1, TLA1, MDT1)
            "",                          # Expected pickup date (blank)
            "",                          # Expected Delivery Date (blank)
            "",                          # Buyer Code (blank)
            "",                          # FOB code (blank)
            "",                          # Line_number (blank)
            row["product_part_number"],  # Item_ID (SKU)
            "EA",                        # Supplier UOM
            row["demand_eaches"],        # Quantity
            "",                          # Shipping Agent Code (blank)
            "",                          # Shipping Agent Name (blank)
            "",                          # Tracking Number (blank)
            "",                          # Closed for Finance Indicator (blank)
            "",                          # Supplier Acknowledgement (blank)
            "",                          # Import Type (blank)
            "",                          # EDI 850 or Email Exempt (blank)
            "",                          # Cancellation Reason (blank)
        ]
        data.append(po_row)

    df = pd.DataFrame(data, columns=columns)
    return df


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

    # Pre-build SKU -> ChewySkuState map for tooltips
    sku_info_map = _build_sku_info_map(vendor)

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

        # --- Apply cell-level color highlighting (unmet/exceed) ---
        styled_agg = comp_df_display.style.apply(_style_agg_cells, axis=None)

        st.dataframe(
            styled_agg,
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

            # ---------- Plan detail (container/SKU grid) ----------
            detail_df = _plan_detail_df(plan)

            if detail_df.empty:
                st.info("No rows in this plan.")
            else:
                # Build SKU -> status map from metrics (unmet / exceed / ok)
                status_map: Dict[str, str] = {}
                metrics = plan.metrics or ContainerPlanMetrics()
                for r in metrics.demand_met_by_sku or []:
                    sku = str(r.get("product_part_number", ""))
                    orig = float(r.get("original_demand", 0.0) or 0.0)
                    assigned = float(r.get("assigned_demand", 0.0) or 0.0)
                    if orig > assigned:
                        status_map[sku] = "unmet"
                    elif assigned > orig:
                        status_map[sku] = "exceed"

                def _style_plan_detail_rows(row: pd.Series):
                    """
                    For any SKU row:
                      - if demand_unmet -> color row red
                      - if demand_exceed -> color row orange
                    """
                    sku = str(row.get("SKU", ""))
                    status = status_map.get(sku, "")

                    base_styles = [""] * len(row)

                    if status == "unmet":
                        color_str = "background-color: #b71c1c; color: white;"
                        return [color_str] * len(row)
                    elif status == "exceed":
                        color_str = "background-color: #e65100; color: white;"
                        return [color_str] * len(row)

                    return base_styles

                # Note: Streamlit's st.dataframe() does NOT support Pandas set_tooltips()
                # So we apply row styling only, and provide SKU details via a selectbox below
                styled_detail = (
                    detail_df.style
                    .apply(_style_plan_detail_rows, axis=1)
                )

                st.markdown("#### Container / SKU Assignment")
                st.caption("💡 Click on a row to filter SKU Details below to that SKU")
                
                # Use dataframe with selection to capture clicked SKU
                selection = st.dataframe(
                    styled_detail,
                    use_container_width=True,
                    height=400,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"container_sku_grid_{label}",
                )
                
                # Extract selected SKU from the selection
                selected_sku_from_grid = None
                if selection and selection.selection and selection.selection.rows:
                    selected_row_idx = selection.selection.rows[0]
                    if selected_row_idx < len(detail_df):
                        selected_sku_from_grid = str(detail_df.iloc[selected_row_idx]["SKU"])

                # ---- SKU Details Grid ----
                unique_skus = detail_df["SKU"].astype(str).unique().tolist()
                skus_with_info = [s for s in unique_skus if s in sku_info_map]

                if skus_with_info:
                    st.markdown("##### SKU Details")
                    
                    sku_details_df = _build_sku_details_df(skus_with_info, sku_info_map)
                    
                    # Session state key for the SKU filter
                    filter_key = f"sku_filter_{label}"
                    
                    # Initialize session state if not exists
                    if filter_key not in st.session_state:
                        st.session_state[filter_key] = ""
                    
                    # If a SKU was selected from the grid above, update session state
                    # (must be done before the widget is rendered)
                    if selected_sku_from_grid:
                        st.session_state[filter_key] = selected_sku_from_grid
                    
                    # Callback function for clear button (runs before widget instantiation on next rerun)
                    def clear_filter_callback(key):
                        st.session_state[key] = ""
                    
                    # Filter controls row
                    filter_col1, filter_col2 = st.columns([4, 1])
                    
                    with filter_col1:
                        # Add search/filter capability
                        sku_search = st.text_input(
                            "🔍 Filter SKUs",
                            key=filter_key,
                            placeholder="Type to filter by SKU or Product Name..."
                        )
                    
                    with filter_col2:
                        # Clear filter button with callback
                        st.button(
                            "Clear Filter", 
                            key=f"clear_filter_{label}",
                            on_click=clear_filter_callback,
                            args=(filter_key,)
                        )
                    
                    if sku_search:
                        mask = (
                            sku_details_df["SKU"].str.contains(sku_search, case=False, na=False) |
                            sku_details_df["Product Name"].str.contains(sku_search, case=False, na=False)
                        )
                        filtered_sku_df = sku_details_df[mask]
                    else:
                        filtered_sku_df = sku_details_df
                    
                    st.dataframe(
                        filtered_sku_df,
                        use_container_width=True,
                        height=400,
                    )
                    
                    # CSV download for SKU details
                    csv_sku_data = sku_details_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download SKU Details as CSV",
                        data=csv_sku_data,
                        file_name=f"{vendor.vendor_Code}_{label}_sku_details.csv",
                        mime="text/csv",
                        key=f"sku_csv_{label}",
                    )

            # ---------- PO Line Items grid + CSV export ----------
            st.markdown("#### PO Line Items")

            po_df = _build_po_lines_df(plan, vendor)

            if po_df.empty:
                st.info("No PO line items for this plan.")
            else:
                st.dataframe(
                    po_df,
                    use_container_width=True,
                )

                csv_data = po_df.to_csv(index=False)
                st.download_button(
                    label="Download PO Lines as CSV",
                    data=csv_data,
                    file_name=f"{vendor.vendor_Code}_{label}_po_lines.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
