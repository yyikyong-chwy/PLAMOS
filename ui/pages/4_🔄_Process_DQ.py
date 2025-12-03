"""
Streamlit page for processing demand data and viewing DQ results.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from io import BytesIO

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import data.sql_lite_store as sql_lite_store
from preprocessing.data_preprocessing import process_demand_data

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


if __name__ == "__main__":
    main()

