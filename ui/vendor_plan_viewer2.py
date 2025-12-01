# ui/app.py

import streamlit as st
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # one level up from ui/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data.vendor_state_loader import load_all_vendor_states
from states.vendorState import vendorState
from states.ContainerPlanMetrics import ContainerPlanMetrics


# ------------- Data loading (cached) ------------- #

@st.cache_data
def load_vendor_states_list():
    """
    Load all vendor states and return as a list of (vendor_code, vendorState).
    """
    states = list(load_all_vendor_states(as_list=True))
    return states


def get_vendor_display_name(vendor: vendorState) -> str:
    name = vendor.vendor_name or ""
    code = vendor.vendor_Code
    return f"{code} - {name}" if name else code


# ------------- UI helpers: Container utilization ------------- #

def build_container_df(metrics: ContainerPlanMetrics, cbm_max: float) -> pd.DataFrame | None:
    """
    Build a normalized container-level dataframe for a plan with columns:
      DEST, container, util, cbm_used

    Utilization is in [0, 1]. If cbm_used is not present, infer from util * cbm_max.
    """
    records = metrics.total_cbm_used_by_container_dest or []
    if not records:
        return None

    df = pd.DataFrame(records)
    if df.empty:
        return None

    # Ensure columns exist
    if "DEST" not in df.columns:
        df["DEST"] = ""
    if "container" not in df.columns:
        df["container"] = 0

    df["DEST"] = df["DEST"].astype(str)
    df["container"] = df["container"].astype(int)

    # Compute util
    if "util" not in df.columns:
        if "cbm_used" in df.columns:
            df["util"] = df["cbm_used"].astype(float) / float(cbm_max or 66.0)
        else:
            df["util"] = 0.0
    else:
        df["util"] = df["util"].astype(float)

    # Compute cbm_used
    if "cbm_used" not in df.columns:
        df["cbm_used"] = df["util"].astype(float) * float(cbm_max or 66.0)
    else:
        df["cbm_used"] = df["cbm_used"].astype(float)

    df = df[["DEST", "container", "util", "cbm_used"]].sort_values(
        ["DEST", "container"]
    )

    return df


def render_container_utilization_chart(
    container_df: pd.DataFrame | None,
    cbm_max: float,
    max_rows: int | None = None,
):
    """
    Render the container utilization area for a single plan.

    - Bars are thick, with no text above.
    - On the right side of each bar: "C1: 52.3 / 66.0".
    - `max_rows` (if provided) is used to pad with empty rows so that
      all plans share the same visual height.
    """
    # Inject CSS for thick bars and side label
    st.markdown(
        """
        <style>
        .container-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }
        .container-bar {
            flex: 1;
        }
        .container-label {
            white-space: nowrap;
            font-size: 0.8rem;
            color: #555;
        }
        .thick-progress-container {
            width: 100%;
            background-color: #f0f2f6;
            border-radius: 8px;
            height: 20px;          /* thickness of bar */
        }
        .thick-progress-fill {
            background-color: #3579f6;
            height: 100%;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if container_df is None or container_df.empty:
        st.info("No container utilization details available for this plan.")
        # still pad with empty rows so height is aligned
        if max_rows is not None:
            for _ in range(max_rows):
                st.markdown(
                    """
                    <div class="container-row">
                        <div class="container-bar">
                            <div class="thick-progress-container">
                                <div class="thick-progress-fill" style="width: 0%; opacity: 0.0;"></div>
                            </div>
                        </div>
                        <div class="container-label">&nbsp;</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    df = container_df.copy()

    # Count how many actual rows we have
    n_rows = len(df)

    # Group by DEST for small separators
    grouped = list(df.groupby("DEST"))

    for dest, g in grouped:
        st.markdown(f"*Destination: `{dest}`*")
        for _, row in g.iterrows():
            cid = int(row["container"])
            util = float(row["util"])
            util_clamped = min(max(util, 0.0), 1.0)
            cbm_used = float(row["cbm_used"])
            label_text = f"C{cid}: {cbm_used:.1f} / {cbm_max:.1f}"

            st.markdown(
                f"""
                <div class="container-row">
                    <div class="container-bar">
                        <div class="thick-progress-container">
                            <div class="thick-progress-fill" style="width: {util_clamped*100:.1f}%;"></div>
                        </div>
                    </div>
                    <div class="container-label">{label_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Pad with empty rows to align height with other plans
    if max_rows is not None:
        remaining = max_rows - n_rows
        for _ in range(max(0, remaining)):
            st.markdown(
                """
                <div class="container-row">
                    <div class="container-bar">
                        <div class="thick-progress-container">
                            <div class="thick-progress-fill" style="width: 0%; opacity: 0.0;"></div>
                        </div>
                    </div>
                    <div class="container-label">&nbsp;</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------- UI helpers: SKU tables ------------- #

def build_sku_table_df(metrics: ContainerPlanMetrics) -> pd.DataFrame | None:
    """
    Build a dataframe for the SKU table with columns:
      SKU, original_demand, assigned_demand, delta

    - Filters out rows where both original_demand and assigned_demand are 0.
    - Casts demands to integers.
    """
    rows = metrics.demand_met_by_sku or []
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Ensure required columns
    if "product_part_number" not in df.columns:
        df["product_part_number"] = ""
    if "original_demand" not in df.columns:
        df["original_demand"] = 0
    if "assigned_demand" not in df.columns:
        df["assigned_demand"] = 0

    # Cast to ints (demands are integers only)
    df["original_demand"] = df["original_demand"].fillna(0).astype(int)
    df["assigned_demand"] = df["assigned_demand"].fillna(0).astype(int)

    # Remove rows where both are 0
    mask = ~((df["original_demand"] == 0) & (df["assigned_demand"] == 0))
    df = df.loc[mask].copy()
    if df.empty:
        return None

    # Delta for coloring only
    df["delta"] = df["assigned_demand"] - df["original_demand"]

    display_df = df.rename(columns={"product_part_number": "SKU"})[
        ["SKU", "original_demand", "assigned_demand", "delta"]
    ]
    return display_df


def render_sku_demand_table(table_df: pd.DataFrame | None, height: int | None = None):
    """
    Render SKU table with:
      - Columns: SKU, original_demand, assigned_demand
      - Index hidden
      - Integer display
      - Row color by delta (assigned - original):
          >0 = blue, <0 = red, 0 = default
    """
    if table_df is None or table_df.empty:
        st.info("No SKU-level demand metrics available for this plan.")
        return

    def color_rows(row):
        d = row["delta"]
        if d > 0:
            # 3 visible cols + hidden delta col
            return ["color: blue;"] * 3 + [""]
        elif d < 0:
            return ["color: red;"] * 3 + [""]
        else:
            return [""] * 4

    styled = (
        table_df.style
        .apply(color_rows, axis=1)
        .hide(axis="columns", names=["delta"])
        .format({
            "original_demand": "{:.0f}",
            "assigned_demand": "{:.0f}",
        })
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


# ------------- UI: Plan column ------------- #

def render_plan_column(
    plan_index: int,
    plan,
    vendor_cbm_max: float,
    container_df: pd.DataFrame | None,
    container_max_rows: int | None,
    sku_table_df: pd.DataFrame | None,
    table_height: int | None,
):
    """
    Renders a full column for a single plan: container utilization + SKU tradeoffs.
    """
    metrics: ContainerPlanMetrics = plan.metrics or ContainerPlanMetrics()

    strategy = getattr(plan, "strategy", "")
    strat_label = getattr(strategy, "value", strategy)  # enum or str
    st.markdown(
        f"#### Plan {plan_index + 1}<br/><small>Strategy: `{strat_label}`</small>",
        unsafe_allow_html=True,
    )

    st.markdown("**Container Utilization**")
    render_container_utilization_chart(
        container_df=container_df,
        cbm_max=float(vendor_cbm_max or 66.0),
        max_rows=container_max_rows,
    )

    st.markdown("---")
    st.markdown("**SKU Assignment Tradeoffs**")
    render_sku_demand_table(sku_table_df, height=table_height)


# ------------- Main app layout ------------- #

def main():
    st.set_page_config(page_title="Vendor Container Plan Explorer", layout="wide")

    st.title("Vendor Container Plan Explorer")
    st.caption("Browse vendor container plans, utilization, and SKU-level tradeoffs.")

    # --- Load vendor states ---
    vendor_states = load_all_vendor_states(as_list=True)
    if not vendor_states:
        st.error("No vendor states found.")
        st.stop()

    vendor_labels = []
    vendors_by_label = {}
    for _, vs in vendor_states:
        if not isinstance(vs, vendorState):
            continue
        label = get_vendor_display_name(vs)
        vendor_labels.append(label)
        vendors_by_label[label] = vs

    # --- Vendor selection ---
    st.sidebar.header("Vendor Selection")
    selected_label = st.sidebar.selectbox("Select vendor", options=sorted(vendor_labels))
    current_vendor: vendorState = vendors_by_label[selected_label]

    st.subheader(f"Vendor: {current_vendor.vendor_Code} – {current_vendor.vendor_name}")
    st.write(f"CBM_Max: **{current_vendor.CBM_Max}**")

    plans = current_vendor.container_plans or []
    total_plans = len(plans)
    if total_plans == 0:
        st.warning("This vendor has no container plans.")
        st.stop()

    st.markdown("### Plan Comparison (3 at a time)")

    # --- Plan selection for 3 slots ---
    st.sidebar.header("Plan Slots")

    plan_labels = [f"Plan {i+1}" for i in range(total_plans)]
    default_indices = list(range(min(3, total_plans)))

    # Primary plan (Column 1)
    primary_idx = st.sidebar.selectbox(
        "Primary Plan (Column 1)",
        options=list(range(total_plans)),
        format_func=lambda i: plan_labels[i],
        index=default_indices[0],
    )

    # Second plan (Column 2) - exclude primary
    remaining_for_second = [i for i in range(total_plans) if i != primary_idx]
    second_idx = None
    if len(remaining_for_second) > 0:
        second_idx = st.sidebar.selectbox(
            "Compare Plan (Column 2)",
            options=remaining_for_second,
            format_func=lambda i: plan_labels[i],
            index=0 if len(default_indices) > 1 and default_indices[1] in remaining_for_second else 0,
        )

    # Third plan (Column 3) - exclude primary + second
    third_idx = None
    if second_idx is not None:
        remaining_for_third = [i for i in remaining_for_second if i != second_idx]
        if len(remaining_for_third) > 0:
            third_idx = st.sidebar.selectbox(
                "Compare Plan (Column 3)",
                options=remaining_for_third,
                format_func=lambda i: plan_labels[i],
                index=0 if len(default_indices) > 2 and default_indices[2] in remaining_for_third else 0,
            )

    selected_plan_indices = [primary_idx]
    if second_idx is not None:
        selected_plan_indices.append(second_idx)
    if third_idx is not None:
        selected_plan_indices.append(third_idx)

    # --- Build SKU tables & container dfs; compute max rows / heights ---
    sku_tables: dict[int, pd.DataFrame | None] = {}
    container_dfs: dict[int, pd.DataFrame | None] = {}

    max_sku_rows = 0
    max_container_rows = 0

    for idx in selected_plan_indices:
        plan = plans[idx]
        metrics: ContainerPlanMetrics = plan.metrics or ContainerPlanMetrics()

        # SKU table
        df_sku = build_sku_table_df(metrics)
        sku_tables[idx] = df_sku
        if df_sku is not None:
            max_sku_rows = max(max_sku_rows, len(df_sku))

        # Container df
        df_cont = build_container_df(metrics, cbm_max=float(current_vendor.CBM_Max or 66.0))
        container_dfs[idx] = df_cont
        if df_cont is not None:
            max_container_rows = max(max_container_rows, len(df_cont))

    # Common table height across visible plans (SKU table)
    row_height = 28   # px per row (approx)
    base_height = 60  # header + padding
    table_height = base_height + row_height * max_sku_rows if max_sku_rows > 0 else None

    # --- Display selected plans in columns ---
    n_cols = len(selected_plan_indices)
    cols = st.columns(n_cols, gap="large")

    for col, plan_idx in zip(cols, selected_plan_indices):
        with col:
            plan = plans[plan_idx]
            render_plan_column(
                plan_index=plan_idx,
                plan=plan,
                vendor_cbm_max=current_vendor.CBM_Max,
                container_df=container_dfs.get(plan_idx),
                container_max_rows=max_container_rows if max_container_rows > 0 else None,
                sku_table_df=sku_tables.get(plan_idx),
                table_height=table_height,
            )


if __name__ == "__main__":
    main()
