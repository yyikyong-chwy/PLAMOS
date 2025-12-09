import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# project imports
from data.vendor_state_store import load_vendor_state_from_db
from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerPlanMetrics import ContainerPlanMetrics
from states.ChewySkuState import ChewySkuState
from agents.planEvalAgent import calculate_revised_projections


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
    threshold: float = 0.95,) -> int:
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


# Location coordinates for distribution centers
DC_LOCATIONS = {
    "MDT1": {"name": "Middletown, PA", "lat": 40.20, "lon": -76.73},
    "TNY1": {"name": "New York, NY", "lat": 40.71, "lon": -74.01},
    "TLA1": {"name": "Los Angeles, CA", "lat": 34.05, "lon": -118.24},
}


def _compute_container_utilization_by_dest(
    plan: ContainerPlanState,
    vendor: vendorState,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute container utilization grouped by destination (DEST).
    Returns a dict: DEST -> list of {container, utilization_pct, cbm_used, cbm_max}
    """
    df = plan.to_df()
    if df.empty:
        return {}

    df = df.copy()
    df["cbm_assigned"] = pd.to_numeric(df["cbm_assigned"], errors="coerce").fillna(0.0)

    cbm_max = float(getattr(vendor, "CBM_Max", 66.0) or 66.0)

    # Group by container and DEST, sum CBM
    container_cbm = (
        df.groupby(["container", "DEST"])["cbm_assigned"]
        .sum()
        .reset_index()
    )

    result: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in container_cbm.iterrows():
        dest = str(row["DEST"])
        container = str(row["container"])
        cbm_used = float(row["cbm_assigned"])
        util_pct = (cbm_used / cbm_max) * 100.0 if cbm_max > 0 else 0.0

        if dest not in result:
            result[dest] = []
        result[dest].append({
            "container": container,
            "utilization_pct": util_pct,
            "cbm_used": cbm_used,
            "cbm_max": cbm_max,
        })

    # Sort containers within each destination
    for dest in result:
        result[dest] = sorted(result[dest], key=lambda x: x["container"])

    return result


def _build_container_map_with_bars(
    utilization_by_dest: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    """
    Build a Plotly figure showing US map with DC locations and container utilization bars
    positioned directly next to each location on the map.
    """
    # Paper coordinates for bar chart positions (x, y) relative to each DC
    # These position the bar charts near each location on the Albers USA projection
    DC_BAR_POSITIONS = {
        "TLA1": {"x": 0.08, "y": 0.25, "anchor": "left"},    # Los Angeles - left side
        "MDT1": {"x": 0.78, "y": 0.65, "anchor": "left"},    # Middletown PA - right side
        "TNY1": {"x": 0.85, "y": 0.45, "anchor": "left"},    # New York - right side
    }
    
    # Calculate number of subplots needed (1 for map + 1 per DC with data)
    dests_with_data = [d for d in DC_LOCATIONS.keys() if d in utilization_by_dest and utilization_by_dest[d]]
    
    # Color function for utilization
    def _util_color(pct: float) -> str:
        if pct >= 95:
            return "#2e7d32"  # Dark green - excellent
        elif pct >= 80:
            return "#66bb6a"  # Light green - good
        elif pct >= 60:
            return "#ffa726"  # Orange - moderate
        else:
            return "#ef5350"  # Red - low

    # Create figure with subplots - map in center, bar charts positioned around it
    # Use domain-based positioning for bar charts
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scattergeo"}]],
    )

    # Configure the geo map
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showlakes=True,
        lakecolor="rgb(255, 255, 255)",
        showcoastlines=True,
        coastlinecolor="rgb(180, 180, 180)",
        showsubunits=True,
        subunitcolor="rgb(200, 200, 200)",
    )

    # Add DC markers on the map
    for dest, info in DC_LOCATIONS.items():
        containers = utilization_by_dest.get(dest, [])
        num_containers = len(containers)
        
        if containers:
            avg_util = sum(c["utilization_pct"] for c in containers) / len(containers)
        else:
            avg_util = 0

        # Build hover text
        if containers:
            hover_lines = [f"<b>{dest}</b> - {info['name']}<br>"]
            hover_lines.append(f"Containers: {num_containers}<br>")
            hover_lines.append(f"Avg Utilization: {avg_util:.1f}%<br>")
            hover_text = "".join(hover_lines)
        else:
            hover_text = f"<b>{dest}</b> - {info['name']}<br>No containers"

        fig.add_trace(go.Scattergeo(
            lon=[info["lon"]],
            lat=[info["lat"]],
            mode="markers+text",
            marker=dict(
                size=18,
                color=_util_color(avg_util) if containers else "#9e9e9e",
                line=dict(width=2, color="white"),
                symbol="circle",
            ),
            text=f"<b>{dest}</b>",
            textposition="top center",
            textfont=dict(size=11, color="#333", family="Arial Black"),
            hovertemplate=hover_text + "<extra></extra>",
            name=dest,
        ))

    # Add bar charts as separate xy traces with specific domains
    # We'll create bar-like visualizations using shapes and annotations
    shapes = []
    annotations = []
    
    bar_height = 0.022  # Height of each bar
    bar_max_width = 0.12  # Maximum width for 100% utilization
    bar_spacing = 0.005  # Space between bars
    
    for dest in DC_LOCATIONS.keys():
        containers = utilization_by_dest.get(dest, [])
        if not containers:
            continue
            
        pos = DC_BAR_POSITIONS[dest]
        base_x = pos["x"]
        base_y = pos["y"]
        
        # Add destination label
        annotations.append(dict(
            x=base_x,
            y=base_y + len(containers) * (bar_height + bar_spacing) + 0.02,
            xref="paper",
            yref="paper",
            text=f"<b>{dest}</b>",
            showarrow=False,
            font=dict(size=11, color="#333", family="Arial Black"),
            xanchor="left",
        ))
        
        # Add bars for each container
        for i, container in enumerate(containers[:8]):  # Limit to 8 containers for space
            util_pct = container["utilization_pct"]
            bar_width = (util_pct / 100.0) * bar_max_width
            y_pos = base_y + i * (bar_height + bar_spacing)
            
            # Background bar (100% reference)
            shapes.append(dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=base_x,
                y0=y_pos,
                x1=base_x + bar_max_width,
                y1=y_pos + bar_height,
                fillcolor="rgba(220, 220, 220, 0.5)",
                line=dict(width=1, color="rgba(180, 180, 180, 0.8)"),
            ))
            
            # Filled bar (actual utilization)
            shapes.append(dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=base_x,
                y0=y_pos,
                x1=base_x + bar_width,
                y1=y_pos + bar_height,
                fillcolor=_util_color(util_pct),
                line=dict(width=0),
            ))
            
            # Container label and percentage
            annotations.append(dict(
                x=base_x + bar_max_width + 0.005,
                y=y_pos + bar_height / 2,
                xref="paper",
                yref="paper",
                text=f"{container['container'][-4:]}: {util_pct:.0f}%",
                showarrow=False,
                font=dict(size=8, color="#555"),
                xanchor="left",
                yanchor="middle",
            ))
        
        # Show count if more containers exist
        if len(containers) > 8:
            annotations.append(dict(
                x=base_x,
                y=base_y - 0.02,
                xref="paper",
                yref="paper",
                text=f"+{len(containers) - 8} more",
                showarrow=False,
                font=dict(size=9, color="#777"),
                xanchor="left",
            ))

    # Update layout - disable all interactions to prevent scrolling
    fig.update_layout(
        title=dict(
            text="Container Distribution by Location",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
        shapes=shapes,
        annotations=annotations,
        dragmode=False,  # Disable drag/pan
    )
    
    # Disable geo map interactions (zoom, pan)
    fig.update_geos(
        fitbounds=False,
    )

    return fig


def _build_sku_projection_df(
    plan: ContainerPlanState,
    vendor: vendorState,
    sku_info_map: Dict[str, ChewySkuState],
) -> pd.DataFrame:
    """
    Build a DataFrame with SKU-level projection data for a given plan.
    Uses calculate_revised_projections from planEvalAgent for centralized logic.
    Includes planned demand, assigned demand, OH, OO, lead time projections, etc.
    """
    # Get SKU states list from sku_info_map
    sku_states = list(sku_info_map.values())
    
    if not sku_states:
        return pd.DataFrame()
    
    # Use centralized projection calculation
    proj_df = calculate_revised_projections(plan, sku_states)
    
    if proj_df.empty:
        return pd.DataFrame()
    
    # Get demand metrics from plan
    metrics = plan.metrics or ContainerPlanMetrics()
    demand_by_sku = metrics.demand_met_by_sku or []
    
    # Build maps for planned and assigned demand
    planned_map: Dict[str, float] = {}
    assigned_map: Dict[str, float] = {}
    for r in demand_by_sku:
        sku = str(r.get("product_part_number", ""))
        planned_map[sku] = float(r.get("original_demand", 0.0) or 0.0)
        assigned_map[sku] = float(r.get("assigned_demand", 0.0) or 0.0)
    
    # Add additional columns from ChewySkuState
    records = []
    for _, row in proj_df.iterrows():
        sku = str(row["product_part_number"])
        cs = sku_info_map.get(sku)
        if not cs:
            continue
        
        # Get values from ChewySkuState
        planned_demand = planned_map.get(sku, float(cs.planned_demand or 0.0))
        assigned_demand = assigned_map.get(sku, row["additional_supply_eaches"])
        oh = int(cs.OH or 0)
        oo = int(cs.ost_ord or 0)
        avg_lt = int(cs.AVG_LT or 0)
        
        # Original DOS values from ChewySkuState
        dos_end_lt_days = float(cs.DOS_end_LT_days or 0.0)
        dos_end_lt_plus4w_days = float(cs.DOS_end_LT_plus4w_days or 0.0)
        
        records.append({
            "SKU": sku,
            "ABC": cs.product_abc_code or "",
            "total_planned_demand": planned_demand,
            "total_assigned_demand": assigned_demand,
            "OH": oh,
            "OO": oo,
            "AVG_LT": avg_lt,
            "consumption_within_LT": float(cs.demand_within_LT or 0.0),
            "projected_OH_end_LT": row["original_projected_OH_end_LT"],
            "DOS_end_LT_days": dos_end_lt_days,
            "revised_projected_OH_end_LT": row["revised_projected_OH_end_LT"],
            "revised_DOS_end_LT": row["revised_DOS_end_LT_days"] or 0.0,
            "projected_OH_end_LT_plus4w": row["original_projected_OH_end_LT_plus4w"],
            "DOS_end_LT_plus4w_days": dos_end_lt_plus4w_days,
            "revised_projected_OH_end_LT_plus4w": row["revised_projected_OH_end_LT_plus4w"],
            "revised_DOS_end_LT_plus4w": row["revised_DOS_end_LT_days_plus4w"] or 0.0,
        })
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Format numeric columns
    int_cols = ["total_planned_demand", "total_assigned_demand", "OH", "OO", "AVG_LT"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)
    
    float_cols = [
        "consumption_within_LT", "projected_OH_end_LT", "DOS_end_LT_days", "revised_projected_OH_end_LT",
        "revised_DOS_end_LT", "projected_OH_end_LT_plus4w", "DOS_end_LT_plus4w_days",
        "revised_projected_OH_end_LT_plus4w", "revised_DOS_end_LT_plus4w"
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].round(2)
    
    # Filter to only SKUs with planned or assigned demand > 0
    df = df[(df["total_planned_demand"] > 0) | (df["total_assigned_demand"] > 0)]
    
    if df.empty:
        return pd.DataFrame()
    
    # Create MultiIndex columns for grouped headers
    multi_columns = [
        ("", "SKU"),
        ("", "ABC"),
        ("", "total_planned_demand"),
        ("", "total_assigned_demand"),
        ("", "OH"),
        ("", "OO"),
        ("", "AVG_LT"),
        ("", "consumption_within_LT"),
        ("@LT", "projected_OH"),
        ("@LT", "DOS_days"),
        ("@LT", "revised_projected_OH"),
        ("@LT", "revised_DOS"),
        ("@LT_plus4w", "projected_OH"),
        ("@LT_plus4w", "DOS_days"),
        ("@LT_plus4w", "revised_projected_OH"),
        ("@LT_plus4w", "revised_DOS"),
    ]
    
    # Reorder columns to match the MultiIndex structure
    ordered_cols = [
        "SKU", "ABC", "total_planned_demand", "total_assigned_demand", "OH", "OO", "AVG_LT", "consumption_within_LT",
        "projected_OH_end_LT", "DOS_end_LT_days", "revised_projected_OH_end_LT", "revised_DOS_end_LT",
        "projected_OH_end_LT_plus4w", "DOS_end_LT_plus4w_days", "revised_projected_OH_end_LT_plus4w", "revised_DOS_end_LT_plus4w"
    ]
    df = df[ordered_cols]
    
    # Apply MultiIndex columns
    df.columns = pd.MultiIndex.from_tuples(multi_columns)
    
    return df


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
        db_results = load_vendor_state_from_db()  # returns list of dicts with 'vendor_code', 'vendor_state'
        vendor_pairs = []
        for r in db_results:
            try:
                vs = vendorState.model_validate(r["vendor_state"])
                vendor_pairs.append((r["vendor_code"], vs))
            except Exception as e:
                st.warning(f"Could not load vendor {r.get('vendor_code', 'unknown')}: {e}")

    if not vendor_pairs:
        st.error("No vendor states found in database.")
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
            "total_assigned_demand",
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
            hide_index=True,
        )
    else:
        st.info("No plan metrics available for this vendor yet.")

    st.markdown("---")

    # ---- Build plan labels for tabs (used in multiple sections) ----
    plan_labels = [
        _short_strategy_name(getattr(plan, "strategy", None))
        for plan in vendor.container_plans
    ]

    # ---- Strategy Selection (controls SKU Projection, Plan Details, and PO Line Items) ----
    st.markdown("### SKU Projection Analysis")
    
    # Strategy selection tabs - this controls all sections below
    selected_strategy_idx = st.radio(
        "Select Strategy",
        options=range(len(plan_labels)),
        format_func=lambda i: plan_labels[i],
        horizontal=True,
        key="strategy_selection",
    )
    
    # Get the selected plan and label
    selected_plan = vendor.container_plans[selected_strategy_idx]
    selected_label = plan_labels[selected_strategy_idx]
    
    # Toggle for hiding details
    _, proj_toggle_col = st.columns([3, 1])
    with proj_toggle_col:
        hide_details = st.toggle("Hide Details", value=False, key="proj_hide_details")
    
    # Columns to hide when "Hide Details" is toggled
    cols_to_hide = [
        ("", "OH"),
        ("", "OO"),
        ("", "AVG_LT"),
        ("", "consumption_within_LT"),
        ("@LT", "projected_OH"),
        ("@LT", "revised_projected_OH"),
        ("@LT_plus4w", "projected_OH"),
        ("@LT_plus4w", "revised_projected_OH"),
    ]
    
    # Columns to format with 2 decimal places
    float_format_cols = [
        ("", "consumption_within_LT"),
        ("@LT", "projected_OH"),
        ("@LT", "DOS_days"),
        ("@LT", "revised_projected_OH"),
        ("@LT", "revised_DOS"),
        ("@LT_plus4w", "projected_OH"),
        ("@LT_plus4w", "DOS_days"),
        ("@LT_plus4w", "revised_projected_OH"),
        ("@LT_plus4w", "revised_DOS"),
    ]
    
    def _style_projection_rows(row: pd.Series) -> List[str]:
        """
        Style rows based on demand comparison:
        - Red if assigned < planned (unmet demand)
        - Orange if assigned > planned (exceed demand)
        """
        # Access MultiIndex columns
        planned = row.get(("", "total_planned_demand"), 0)
        assigned = row.get(("", "total_assigned_demand"), 0)
        
        num_cols = len(row)
        
        if assigned < planned:
            return ["background-color: #b71c1c; color: white;"] * num_cols
        elif assigned > planned:
            return ["background-color: #e65100; color: white;"] * num_cols
        else:
            return [""] * num_cols
    
    # Show SKU Projection for selected strategy
    projection_df = _build_sku_projection_df(selected_plan, vendor, sku_info_map)
    
    if projection_df.empty:
        st.info("No SKU projection data available for this plan.")
    else:
        # Filter columns if hide_details is enabled
        display_df = projection_df.copy()
        if hide_details:
            cols_to_keep = [c for c in display_df.columns if c not in cols_to_hide]
            display_df = display_df[cols_to_keep]
        
        # Build format dict for 2 decimal display on @LT columns
        format_dict = {col: "{:.2f}" for col in float_format_cols if col in display_df.columns}
        
        # Apply row styling and number formatting
        styled_projection = (
            display_df.style
            .apply(_style_projection_rows, axis=1)
            .format(format_dict)
        )
        
        st.dataframe(
            styled_projection,
            use_container_width=True,
            height=400,
            hide_index=True,
        )
        
        # CSV download for projection data (always full data)
        csv_proj_data = projection_df.to_csv(index=False)
        st.download_button(
            label="📥 Download SKU Projections as CSV",
            data=csv_proj_data,
            file_name=f"{vendor.vendor_Code}_{selected_label}_sku_projections.csv",
            mime="text/csv",
            key=f"proj_csv_{selected_label}",
        )

    st.markdown("---")

    # ---- Container Plan Map (uses selected strategy) ----
    st.markdown(f"### Container Plan — {selected_label}")
    
    # Compute container utilization by destination
    utilization_by_dest = _compute_container_utilization_by_dest(selected_plan, vendor)
    
    if not utilization_by_dest:
        st.info("No container assignments available for this plan.")
    else:
        # Display the US map with container bars positioned near each DC
        map_fig = _build_container_map_with_bars(utilization_by_dest)
        st.plotly_chart(
            map_fig,
            use_container_width=True,
            config={
                "scrollZoom": False,
                "doubleClick": False,
                "displayModeBar": False,
            },
        )
        
        # Summary metrics
        total_containers = sum(len(containers) for containers in utilization_by_dest.values())
        all_utils = [c["utilization_pct"] for containers in utilization_by_dest.values() for c in containers]
        avg_util = sum(all_utils) / len(all_utils) if all_utils else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Containers", total_containers)
        with col2:
            st.metric("Avg Utilization", f"{avg_util:.1f}%")
        with col3:
            high_util = sum(1 for u in all_utils if u >= 80)
            st.metric("Containers ≥80% Full", high_util)

    st.markdown("---")

    # ---- Plan Details (uses selected strategy from above) ----
    st.markdown(f"### Plan Details — {selected_label}")

    # ---------- Plan detail (container/SKU grid) ----------
    detail_df = _plan_detail_df(selected_plan)

    if detail_df.empty:
        st.info("No rows in this plan.")
    else:
        # Build SKU -> status map from metrics (unmet / exceed / ok)
        status_map: Dict[str, str] = {}
        metrics = selected_plan.metrics or ContainerPlanMetrics()
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
        
        st.dataframe(
            styled_detail,
            use_container_width=True,
            height=400,
        )

    # ---------- PO Line Items grid + CSV export ----------
    st.markdown("#### PO Line Items")

    po_df = _build_po_lines_df(selected_plan, vendor)

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
            file_name=f"{vendor.vendor_Code}_{selected_label}_po_lines.csv",
            mime="text/csv",
            key="po_lines_csv",
        )


if __name__ == "__main__":
    main()
