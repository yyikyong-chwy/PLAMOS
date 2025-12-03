"""
Streamlit page for processing demand data and viewing DQ results.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from io import BytesIO, StringIO
import contextlib

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import data.sql_lite_store as sql_lite_store
from preprocessing.data_preprocessing import process_demand_data
from agents.ContainerPlanningOrchestrator import (
    load_data, 
    generate_vendor_states, 
    compile_app,
)
from states.vendorState import vendorState
import data.vendor_state_store as vendor_state_store

st.set_page_config(
    page_title="Process DQ & Run",
    page_icon="🔄",
    layout="wide",
)


def export_df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Export DataFrame to Excel bytes for download."""
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return out.getvalue()


def main():
    st.title("🔄 Process DQ & Run")
    st.markdown("Process demand data and review Data Quality checks")
    
    st.divider()
    
    # --- Process Section ---
    st.subheader("Run Data Processing")
    
    st.info(
        "**What this does:**\n"
        "1. Loads demand data from local database\n"
        "2. Fetches Kepler splits and SKU supply snapshot from Snowflake\n"
        "3. Joins and enriches the data\n"
        "4. Saves processed data to `df_sku_data` table\n"
        "5. Identifies SKUs with no matching record in PLM"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button(
            "▶️ Process Demand Data",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Processing demand data... This may take a moment."):
                    df_result = process_demand_data()
                
                if df_result is not None and not df_result.empty:
                    st.success(f"✅ Successfully processed {len(df_result):,} SKU records!")
                    st.session_state["dq_processed"] = True
                else:
                    st.warning("⚠️ Processing completed but no data was returned.")
                    st.session_state["dq_processed"] = True
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.exception(e)
    
    with col2:
        st.markdown("**Prerequisites:**")
        st.caption("• `demand_data` table must be uploaded")
        st.caption("• `CBM_Max` table must be uploaded")
        st.caption("• Snowflake connection must be configured (`.env` file)")
    
    st.divider()
    
    # --- DQ Results Section ---
    st.subheader("📋 Data Quality Check: SKUs with No Record in PLM")
    st.markdown(
        "The table below shows SKUs from the demand data that do not have a matching record "
        "in the PLM (vendor_cbm) data. These SKUs may need attention."
    )
    
    # Load DQ table
    df_dq = sql_lite_store.load_table("DQ_skus_w_no_record_in_plm")
    
    if df_dq.empty:
        st.info(
            "No DQ results available yet. Click **Process Demand Data** above to generate results, "
            "or the DQ check found no issues (all SKUs matched)."
        )
    else:
        # Metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Unmatched SKUs", f"{len(df_dq):,}")
        with metric_cols[1]:
            total_demand = df_dq["Planned_Demand"].sum() if "Planned_Demand" in df_dq.columns else 0
            st.metric("Total Planned Demand", f"{total_demand:,.0f}")
        with metric_cols[2]:
            st.metric("Columns", f"{len(df_dq.columns)}")
        with metric_cols[3]:
            st.metric("Status", "⚠️ Needs Review" if len(df_dq) > 0 else "✅ OK")
        
        # Display table
        st.dataframe(
            df_dq,
            use_container_width=True,
            height=400
        )
        
        # Column details expander
        with st.expander("📋 Column Details"):
            col_info = pd.DataFrame({
                "Column": df_dq.columns,
                "Data Type": [str(df_dq[c].dtype) for c in df_dq.columns],
                "Non-Null Count": [df_dq[c].notna().sum() for c in df_dq.columns],
                "Null Count": [df_dq[c].isna().sum() for c in df_dq.columns],
                "Sample Value": [
                    str(df_dq[c].dropna().iloc[0]) if df_dq[c].notna().any() else "N/A" 
                    for c in df_dq.columns
                ]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Export option
        st.divider()
        xlsx_bytes = export_df_to_xlsx_bytes(df_dq, sheet_name="DQ_SKUs_No_PLM")
        st.download_button(
            "📥 Export DQ Results to Excel",
            xlsx_bytes,
            file_name="DQ_skus_w_no_record_in_plm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.divider()
    
    # --- Container Planning Workflow Section ---
    st.subheader("🚀 Run Container Planning Workflow")
    st.markdown(
        "Execute the container planning optimization workflow for all vendors. "
        "This runs the LangGraph agent pipeline to generate optimal container plans."
    )
    
    st.info(
        "**What this does:**\n"
        "1. Loads processed SKU data from database\n"
        "2. Generates vendor states with demand by destination\n"
        "3. Runs the planning workflow for each vendor\n"
        "4. Saves optimized container plans"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button(
            "🚀 Run Container Planning",
            type="primary",
            use_container_width=True,
            key="run_workflow_btn"
        ):
            try:
                # Status placeholder for live updates
                status_container = st.empty()
                progress_bar = st.progress(0)
                log_container = st.empty()
                
                status_container.info("📊 Loading data...")
                
                # Load data
                df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest = load_data()
                status_container.info("🔧 Generating vendor states...")
                
                # Generate vendor states
                vendor_state_list = generate_vendor_states(
                    df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest
                )
                
                total_vendors = len(vendor_state_list)
                status_container.info(f"🏭 Processing {total_vendors} vendors...")
                
                # Collect all logs
                all_logs = []
                results_summary = []
                
                for idx, current_vendor_state in enumerate(vendor_state_list):
                    vendor_code = current_vendor_state.vendor_Code
                    vendor_name = current_vendor_state.vendor_name
                    
                    # Update progress
                    progress = (idx + 1) / total_vendors
                    progress_bar.progress(progress)
                    status_container.info(
                        f"🔄 Processing vendor {idx + 1}/{total_vendors}: "
                        f"{vendor_code} - {vendor_name}"
                    )
                    
                    # Capture print output
                    log_buffer = StringIO()
                    with contextlib.redirect_stdout(log_buffer):
                        print(f"\n{'='*50}")
                        print(f"Vendor: {vendor_code} - {vendor_name}")
                        print(f"{'='*50}")
                        
                        config = {
                            "configurable": {"thread_id": f"session_{vendor_code}"},
                            "recursion_limit": 500,
                        }
                        app = compile_app()
                        state = app.invoke(current_vendor_state, config=config)
                        current_vendor_state = vendorState.model_validate(state)
                        
                        # Print scores
                        if current_vendor_state.container_plans:
                            metrics = current_vendor_state.container_plans[0].metrics
                            print(f"\n📈 Scores:")
                            print(f"  Overall Score: {metrics.overall_score:.2f}")
                            print(f"  Utilization: {metrics.avg_utilization:.2%}")
                            print(f"  APE vs Planned: {metrics.ape_vs_planned:.2%}")
                            print(f"  APE vs Base: {metrics.ape_vs_base:.2%}")
                            print(f"  Containers: {metrics.containers}")
                            
                            results_summary.append({
                                "Vendor Code": vendor_code,
                                "Vendor Name": vendor_name,
                                "Overall Score": f"{metrics.overall_score:.2f}",
                                "Utilization": f"{metrics.avg_utilization:.2%}",
                                "APE vs Planned": f"{metrics.ape_vs_planned:.2%}",
                                "APE vs Base": f"{metrics.ape_vs_base:.2%}",
                                "Containers": metrics.containers
                            })
                        
                        # Save vendor state
                        vendor_state_store.save_vendor_state_blob(".", current_vendor_state)
                        print(f"✅ Saved vendor state for {vendor_code}")
                    
                    # Append logs
                    log_output = log_buffer.getvalue()
                    all_logs.append(log_output)
                    
                    # Update log display
                    log_container.code("\n".join(all_logs), language="text")
                
                # Final status
                progress_bar.progress(1.0)
                status_container.success(
                    f"✅ Completed! Processed {total_vendors} vendors successfully."
                )
                
                # Show results summary table
                if results_summary:
                    st.subheader("📊 Results Summary")
                    df_results = pd.DataFrame(results_summary)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during workflow: {e}")
                st.exception(e)
    
    with col2:
        st.markdown("**Prerequisites:**")
        st.caption("• `df_sku_data` must be generated (run Process Demand Data first)")
        st.caption("• `CBM_Max` table must be uploaded")
        st.caption("• `Keppler_Split_Perc` table must exist")


if __name__ == "__main__":
    main()

