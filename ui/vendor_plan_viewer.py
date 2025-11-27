"""
Streamlit App for Viewing Vendor Container Plans
Allows users to load, select, and analyze vendor states and their container plans.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vendor_state_loader import load_all_vendor_states, list_available_vendors
from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
from states.ContainerPlanMetrics import ContainerPlanMetrics

# Page configuration
st.set_page_config(
    page_title="Vendor Container Plan Viewer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_vendors():
    """Load all vendor states from disk"""
    vendors = {}
    for vendor_code, vendor_state in load_all_vendor_states():
        vendors[vendor_code] = vendor_state
    return vendors


def display_metrics_card(metrics: ContainerPlanMetrics, plan_idx: int):
    """Display metrics in a structured card format"""
    
    st.subheader(f"📊 Plan {plan_idx + 1} Metrics")
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{metrics.overall_score:.1f}",
            help="Overall plan quality score (0-100)"
        )
    
    with col2:
        st.metric(
            "Containers",
            metrics.containers,
            help="Total number of containers in plan"
        )
    
    with col3:
        st.metric(
            "Avg Utilization",
            f"{metrics.avg_utilization:.1%}",
            help="Average container utilization"
        )
    
    with col4:
        st.metric(
            "Total CBM Used",
            f"{metrics.total_cbm_used:.2f}",
            help="Total cubic meters used"
        )
    
    # Detailed Metrics in Expandable Sections
    with st.expander("📈 Utilization Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Weighted Utilization", f"{metrics.weighted_utilization:.2%}")
        
        with col2:
            st.metric("Low Util Count", metrics.low_util_count, 
                     help=f"Containers below {metrics.low_util_threshold:.0%} utilization")
        
        with col3:
            st.metric("Total Capacity", f"{metrics.total_cbm_capacity:.2f} CBM")
    
    with st.expander("🎯 Accuracy Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("APE vs Planned", f"{metrics.ape_vs_planned:.2%}",
                     help="Average Percentage Error vs Planned")
        
        with col2:
            st.metric("APE vs Base", f"{metrics.ape_vs_base:.2%}",
                     help="Average Percentage Error vs Base")
        
        with col3:
            st.metric("APE vs Excess", f"{metrics.ape_vs_excess:.2%}",
                     help="Average Percentage Error vs Excess")
    
    with st.expander("⚠️ Issues & Violations"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Demand Shortfall", f"{metrics.demand_shortfall:.2f}")
        
        with col2:
            st.metric("SKU Splits Off MCP", metrics.sku_splits_off_mcp,
                     help="SKU splits not aligned with Master Case Pack")
        
        with col3:
            st.metric("MOQ Violations", metrics.moq_violations,
                     help="Minimum Order Quantity violations")
    
    # Container-specific metrics
    if metrics.total_cbm_used_by_container_dest:
        with st.expander("🚢 Container Breakdown by Destination"):
            df_containers = pd.DataFrame(metrics.total_cbm_used_by_container_dest)
            st.dataframe(df_containers, use_container_width=True)
    
    if metrics.total_excess_in_cbm_by_container:
        with st.expander("📦 Excess CBM by Container"):
            excess_df = pd.DataFrame([
                {"Container": k, "Excess CBM": v} 
                for k, v in metrics.total_excess_in_cbm_by_container.items()
            ])
            st.dataframe(excess_df, use_container_width=True)
    
    # Demand Met metrics
    if metrics.demand_met_by_sku:
        with st.expander("📊 Demand Met by SKU (in eaches)", expanded=False):
            demand_df = pd.DataFrame(metrics.demand_met_by_sku)
            
            # Sort by absolute delta to show biggest differences first
            demand_df['abs_delta'] = demand_df['delta'].abs()
            demand_df = demand_df.sort_values(by='abs_delta', ascending=False)
            demand_df = demand_df.drop(columns=['abs_delta'])
            
            # Add a status column
            def get_status(delta):
                if delta > 0:
                    return "Over-allocated"
                elif delta < 0:
                    return "Under-allocated"
                else:
                    return "Exact match"
            
            demand_df['status'] = demand_df['delta'].apply(get_status)
            
            # Format numbers for display
            display_df = demand_df.copy()
            display_df['original_demand'] = display_df['original_demand'].apply(lambda x: f"{x:.0f}")
            display_df['assigned_demand'] = display_df['assigned_demand'].apply(lambda x: f"{x:.0f}")
            display_df['delta'] = display_df['delta'].apply(lambda x: f"{x:+.0f}")
            
            # Reorder columns
            display_df = display_df[['product_part_number', 'original_demand', 'assigned_demand', 'delta', 'status']]
            display_df.columns = ['SKU', 'Original Demand', 'Assigned Demand', 'Delta', 'Status']
            
            st.dataframe(
                display_df, 
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_original = sum([d['original_demand'] for d in metrics.demand_met_by_sku])
            total_assigned = sum([d['assigned_demand'] for d in metrics.demand_met_by_sku])
            over_allocated_count = sum([1 for d in metrics.demand_met_by_sku if d['delta'] > 0])
            under_allocated_count = sum([1 for d in metrics.demand_met_by_sku if d['delta'] < 0])
            
            with col1:
                st.metric("Total Original Demand", f"{total_original:,.0f}")
            
            with col2:
                st.metric("Total Assigned Demand", f"{total_assigned:,.0f}")
            
            with col3:
                st.metric("Over-allocated SKUs", over_allocated_count)
            
            with col4:
                st.metric("Under-allocated SKUs", under_allocated_count)
    
    if metrics.container_utilization_status_info:
        st.info(f"ℹ️ {metrics.container_utilization_status_info}")
    
    if metrics.notes:
        st.warning(f"📝 Notes: {metrics.notes}")


def create_metrics_comparison_chart(plans: list[ContainerPlanState]):
    """Create comparison charts for multiple plans"""
    
    if len(plans) <= 1:
        return None
    
    # Extract metrics for comparison
    metrics_data = []
    for idx, plan in enumerate(plans):
        # Calculate demand metrics
        total_original_demand = 0
        total_assigned_demand = 0
        if plan.metrics.demand_met_by_sku:
            total_original_demand = sum([d['original_demand'] for d in plan.metrics.demand_met_by_sku])
            total_assigned_demand = sum([d['assigned_demand'] for d in plan.metrics.demand_met_by_sku])
        
        demand_fulfillment_rate = 0
        if total_original_demand > 0:
            demand_fulfillment_rate = (total_assigned_demand / total_original_demand) * 100
        
        metrics_data.append({
            "Plan": f"Plan {idx + 1}",
            "Overall Score": plan.metrics.overall_score,
            "Avg Utilization": plan.metrics.avg_utilization * 100,
            "Containers": plan.metrics.containers,
            "APE vs Planned": plan.metrics.ape_vs_planned * 100,
            "APE vs Base": plan.metrics.ape_vs_base * 100,
            "Total CBM Used": plan.metrics.total_cbm_used,
            "Total Original Demand": total_original_demand,
            "Total Assigned Demand": total_assigned_demand,
            "Demand Fulfillment %": demand_fulfillment_rate,
        })
    
    df_comparison = pd.DataFrame(metrics_data)
    
    return df_comparison


def display_plan_comparison(plans: list[ContainerPlanState]):
    """Display comparison between multiple plans"""
    
    st.subheader("📊 Plan Comparison")
    
    df_comparison = create_metrics_comparison_chart(plans)
    
    if df_comparison is None or len(df_comparison) <= 1:
        st.info("Only one plan available - no comparison needed")
        return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Key Metrics", "📊 Demand Metrics", "🎯 Performance Radar", "📋 Data Table"])
    
    with tab1:
        # Bar charts for key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                df_comparison,
                x="Plan",
                y="Overall Score",
                title="Overall Score Comparison",
                color="Overall Score",
                color_continuous_scale="Viridis"
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True, key="chart_overall_score")
            
            fig3 = px.bar(
                df_comparison,
                x="Plan",
                y="Containers",
                title="Number of Containers",
                color="Containers",
                color_continuous_scale="Blues"
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True, key="chart_containers")
        
        with col2:
            fig2 = px.bar(
                df_comparison,
                x="Plan",
                y="Avg Utilization",
                title="Average Utilization (%)",
                color="Avg Utilization",
                color_continuous_scale="Greens"
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, key="chart_utilization")
            
            fig4 = px.bar(
                df_comparison,
                x="Plan",
                y=["APE vs Planned", "APE vs Base"],
                title="Accuracy Metrics (%)",
                barmode="group"
            )
            st.plotly_chart(fig4, use_container_width=True, key="chart_accuracy")
    
    with tab2:
        # Demand metrics charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_demand = px.bar(
                df_comparison,
                x="Plan",
                y=["Total Original Demand", "Total Assigned Demand"],
                title="Original vs Assigned Demand (eaches)",
                barmode="group",
                color_discrete_sequence=["#636EFA", "#EF553B"]
            )
            st.plotly_chart(fig_demand, use_container_width=True, key="chart_demand_comparison")
        
        with col2:
            fig_fulfillment = px.bar(
                df_comparison,
                x="Plan",
                y="Demand Fulfillment %",
                title="Demand Fulfillment Rate (%)",
                color="Demand Fulfillment %",
                color_continuous_scale="RdYlGn"
            )
            # Add a reference line at 100%
            fig_fulfillment.add_hline(y=100, line_dash="dash", line_color="gray", 
                                     annotation_text="100% Target")
            fig_fulfillment.update_layout(showlegend=False)
            st.plotly_chart(fig_fulfillment, use_container_width=True, key="chart_fulfillment")
    
    with tab3:
        # Radar chart for multi-dimensional comparison
        categories = ["Overall Score", "Avg Utilization", "Demand Fulfillment %"]
        
        fig_radar = go.Figure()
        
        for idx, row in df_comparison.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row["Overall Score"], row["Avg Utilization"], row["Demand Fulfillment %"]],
                theta=categories,
                fill='toself',
                name=row["Plan"]
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Multi-Dimensional Plan Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, key="chart_radar")
    
    with tab4:
        # Data table with highlighting
        st.dataframe(
            df_comparison.style.highlight_max(
                subset=["Overall Score", "Avg Utilization"],
                color="lightgreen"
            ).highlight_min(
                subset=["APE vs Planned", "APE vs Base", "Containers"],
                color="lightblue"
            ),
            use_container_width=True
        )
        
        # Trade-offs analysis
        st.subheader("🔄 Trade-offs Analysis")
        
        best_score_idx = df_comparison["Overall Score"].idxmax()
        best_util_idx = df_comparison["Avg Utilization"].idxmax()
        min_containers_idx = df_comparison["Containers"].idxmin()
        best_demand_idx = df_comparison["Demand Fulfillment %"].idxmax()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success(f"**Best Overall Score:** {df_comparison.loc[best_score_idx, 'Plan']}")
            st.write(f"Score: {df_comparison.loc[best_score_idx, 'Overall Score']:.1f}")
        
        with col2:
            st.success(f"**Best Utilization:** {df_comparison.loc[best_util_idx, 'Plan']}")
            st.write(f"Utilization: {df_comparison.loc[best_util_idx, 'Avg Utilization']:.1f}%")
        
        with col3:
            st.success(f"**Fewest Containers:** {df_comparison.loc[min_containers_idx, 'Plan']}")
            st.write(f"Containers: {df_comparison.loc[min_containers_idx, 'Containers']:.0f}")
        
        with col4:
            st.success(f"**Best Demand Fulfillment:** {df_comparison.loc[best_demand_idx, 'Plan']}")
            st.write(f"Fulfillment: {df_comparison.loc[best_demand_idx, 'Demand Fulfillment %']:.1f}%")


def display_container_plan_details(plan: ContainerPlanState, plan_idx: int):
    """Display detailed information about a container plan"""
    
    st.subheader(f"📦 Plan {plan_idx + 1} Details")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Strategy:** {plan.strategy}")
    
    with col2:
        st.write(f"**Loop Counter:** {plan.plan_loop_counter}")
    
    with col3:
        st.write(f"**Total Rows:** {len(plan.container_plan_rows)}")
    
    # Container plan data
    if plan.container_plan_rows:
        df_plan = plan.to_df()
        
        st.subheader("📋 Container Assignments")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            destinations = ["All"] + sorted(df_plan["DEST"].unique().tolist())
            selected_dest = st.selectbox("Filter by Destination", destinations, key=f"dest_{plan_idx}")
        
        with col2:
            if "container" in df_plan.columns:
                containers = ["All"] + sorted([c for c in df_plan["container"].unique() if pd.notna(c)])
                selected_container = st.selectbox("Filter by Container", containers, key=f"cont_{plan_idx}")
            else:
                selected_container = "All"
        
        with col3:
            search_sku = st.text_input("Search SKU", key=f"sku_{plan_idx}")
        
        # Apply filters
        filtered_df = df_plan.copy()
        
        if selected_dest != "All":
            filtered_df = filtered_df[filtered_df["DEST"] == selected_dest]
        
        if selected_container != "All" and "container" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["container"] == selected_container]
        
        if search_sku:
            filtered_df = filtered_df[
                filtered_df["product_part_number"].str.contains(search_sku, case=False, na=False)
            ]
        
        # Display filtered data
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total SKUs", len(filtered_df))
        
        with col2:
            if "cases_assigned" in filtered_df.columns:
                st.metric("Total Cases", filtered_df["cases_assigned"].sum())
        
        with col3:
            if "cbm_assigned" in filtered_df.columns:
                st.metric("Total CBM", f"{filtered_df['cbm_assigned'].sum():.2f}")
        
        with col4:
            if "container" in filtered_df.columns:
                unique_containers = filtered_df["container"].nunique()
                st.metric("Unique Containers", unique_containers)
        
        # Visualizations
        if "container" in filtered_df.columns and "cbm_assigned" in filtered_df.columns:
            st.subheader("📈 CBM Distribution by Container")
            
            cbm_by_container = filtered_df.groupby("container")["cbm_assigned"].sum().reset_index()
            cbm_by_container.columns = ["Container", "Total CBM"]
            
            fig = px.bar(
                cbm_by_container,
                x="Container",
                y="Total CBM",
                title="CBM Utilization by Container",
                color="Total CBM",
                color_continuous_scale="Blues"
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"chart_cbm_dist_plan_{plan_idx}")


def main():
    """Main Streamlit app"""
    
    st.title("📦 Vendor Container Plan Viewer")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load vendors button
        if st.button("🔄 Reload Vendor Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Load vendor data
        with st.spinner("Loading vendor data..."):
            vendors = load_vendors()
        
        if not vendors:
            st.error("❌ No vendors found in data/vendor_plans/")
            st.stop()
        
        st.success(f"✅ Loaded {len(vendors)} vendors")
        
        # Vendor selection
        st.header("🏢 Select Vendor")
        
        vendor_codes = sorted(vendors.keys())
        
        # Create display names with vendor info
        vendor_options = {}
        for code in vendor_codes:
            vendor = vendors[code]
            display_name = f"{code}"
            if vendor.vendor_name:
                display_name += f" - {vendor.vendor_name}"
            display_name += f" ({vendor.numberofPlans()} plans)"
            vendor_options[display_name] = code
        
        selected_display = st.selectbox(
            "Choose a vendor:",
            list(vendor_options.keys())
        )
        
        selected_vendor_code = vendor_options[selected_display]
        selected_vendor = vendors[selected_vendor_code]
        
        st.markdown("---")
        
        # Vendor info
        st.header("ℹ️ Vendor Info")
        st.write(f"**Code:** {selected_vendor.vendor_Code}")
        st.write(f"**Name:** {selected_vendor.vendor_name or 'N/A'}")
        st.write(f"**CBM Max:** {selected_vendor.CBM_Max}")
        st.write(f"**Number of SKUs:** {len(selected_vendor.ChewySku_info)}")
        st.write(f"**Number of Plans:** {selected_vendor.numberofPlans()}")
    
    # Main content area
    if selected_vendor.numberofPlans() == 0:
        st.warning("⚠️ This vendor has no container plans")
        return
    
    # Display plans comparison if multiple plans exist
    if selected_vendor.numberofPlans() > 1:
        display_plan_comparison(selected_vendor.container_plans)
        st.markdown("---")
    
    # Plan selector
    st.header("📋 Individual Plan Details")
    
    plan_tabs = st.tabs([f"Plan {i+1}" for i in range(selected_vendor.numberofPlans())])
    
    for idx, tab in enumerate(plan_tabs):
        with tab:
            plan = selected_vendor.container_plans[idx]
            
            # Display metrics
            display_metrics_card(plan.metrics, idx)
            
            st.markdown("---")
            
            # Display plan details
            display_container_plan_details(plan, idx)


if __name__ == "__main__":
    main()

