import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

import sys
ROOT = Path(__file__).resolve().parents[1]  # one level up from ui/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, sys

# Redirect stdout/stderr to logs once per session
if "stdout_redirected" not in st.session_state:
    os.makedirs("logs", exist_ok=True)
    st.session_state["stdout_redirected"] = True
    sys.stdout = open("logs/prints.out.log", "a", buffering=1, encoding="utf-8")  # line-buffered
    sys.stderr = open("logs/prints.err.log", "a", buffering=1, encoding="utf-8")

import data.snowflake_pull as snowflake_pull
import data.sql_lite_store as sql_lite_store
import agents.workingFlow as wf


# ---------- Helpers ----------
def get_db_path_from_store():
    """Try to get DB path from sql_lite_store, otherwise fall back to data/inventory_data.db"""
    try:
        return sql_lite_store.LOCAL_DB_PATH  # if exposed by your module
    except Exception:
        return "data/inventory_data.db"

def list_db_tables():
    """List user tables in the SQLite DB."""
    db_path = get_db_path_from_store()
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%' 
            and name not like 'DQ_%'
            and name not like 'output_%'
            and name not like 'intermediate_%'
            ORDER BY name
        """)
        tables = [r[0] for r in cur.fetchall()]
    except Exception:
        tables = []
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return tables

def export_df_to_xlsx_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    out.seek(0)
    return out.read()

def read_excel_visible_sheets(file):
    """Return (sheet_names_visible, {sheet_name: df}) skipping hidden/veryHidden sheets."""
    try:
        from openpyxl import load_workbook
        file.seek(0)
        wb = load_workbook(file, read_only=True, data_only=True)
        visible = [ws.title for ws in wb.worksheets if ws.sheet_state not in ("hidden", "veryHidden")]
        wb.close()
        file.seek(0)
        xls = pd.ExcelFile(file)
        # Only keep sheet names that exist in the file and are visible per openpyxl
        sheet_names = [s for s in xls.sheet_names if s in visible]
        sheet_map = {s: xls.parse(s) for s in sheet_names}
        return sheet_names, sheet_map
    except Exception:
        # Fallback: let pandas decide, no visibility filter
        file.seek(0)
        xls = pd.ExcelFile(file)
        sheet_names = xls.sheet_names
        sheet_map = {s: xls.parse(s) for s in sheet_names}
        return sheet_names, sheet_map


# ---------- Snowflake fetchers ----------
def fetch_keppler_split_perc():
    """
    Pull Keppler split percentage data from Snowflake using your snowflake_pull module.    
    Expects:
      - snowflake_pull.get_snowflake_config()
      - snowflake_pull.setconnection(config)
      - snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)
      - snowflake_pull.mutate_keppler_splits(df)
    """
    try:
        cfg = snowflake_pull.get_snowflake_config()
        conn = snowflake_pull.setconnection(cfg)
        df = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)
        df = snowflake_pull.mutate_keppler_splits(df)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Snowflake error: {e}")
        return pd.DataFrame()


def fetch_vendor_cbm():
    """
    Pull vendor CBM data from Snowflake using your snowflake_pull module.
    """
    try:
        cfg = snowflake_pull.get_snowflake_config()
        conn = snowflake_pull.setconnection(cfg)
        df = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_Vendor_CBM)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Snowflake error: {e}")
        return pd.DataFrame()


# ---------- UI ----------
def main():
    st.set_page_config(page_title="Vendor-Container Manager", page_icon="📦", layout="wide")
    st.title("📦 Vendor-Container Manager")

    # Ensure DB exists/ready
    sql_lite_store.init_database()

    # ======================= TOP-LEVEL TABS =======================
    tab_input, tab_dq, tab_output = st.tabs(["🧩 Input", "🧪 DQ", "📤 Output"])

    # ----------------------- INPUT: Sub-tabs -----------------------
    with tab_input:
        st.caption("Ingest data, fetch from Snowflake, and review/export your local SQLite tables.")
        input_tab1, input_tab2, input_tab3 = st.tabs([
            "📤 Upload Excel into local DB",
            "❄️ Fetch from Snowflake and Save",
            "📊 Browse & Export local DB Tables",
        ])

        # ===== Input Sub-tab 1: Upload Excel =====
        with input_tab1:
            st.subheader("Upload Excel into Local DB")
            st.caption("Choose which table to save to, preview the data, and persist it locally.")

            excel_targets = ["demand_data", "CBM_Max"]
            target_table = st.selectbox("Target table to save:", excel_targets, index=0, key="upload_target_table")

            up = st.file_uploader("Upload .xlsx file", type=["xlsx"], key="upload_xlsx")
            if up:
                sheet_names, sheet_map = read_excel_visible_sheets(up)
                if not sheet_names:
                    st.error("No sheets detected.")
                else:
                    st.success(f"Found {len(sheet_names)} visible sheet(s).")
                    chosen_sheet = st.selectbox("Select sheet to preview:", sheet_names, index=0, key="upload_sheet_select")
                    df_preview = sheet_map[chosen_sheet]

                    st.write(f"**Preview — {chosen_sheet}** ({len(df_preview):,} rows × {len(df_preview.columns):,} cols)")
                    st.dataframe(df_preview.head(100), use_container_width=True, height=420)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Rows", f"{len(df_preview):,}")
                    with c2:
                        st.metric("Columns", f"{len(df_preview.columns):,}")
                    with c3:
                        st.metric("Target Table", target_table)

                    if st.button("💾 Save to Database", type="primary", key="btn_save_upload"):
                        try:
                            ok, count = sql_lite_store.save_table(df_preview, target_table)
                            if ok:
                                st.success(f"Saved {count} row(s) into '{target_table}'.")
                                st.balloons()
                            else:
                                st.error("Save failed (save_table returned False).")
                        except Exception as e:
                            st.error(f"Save failed: {e}")

        # ===== Input Sub-tab 2: Snowflake Fetch =====
        with input_tab2:
            st.subheader("Fetch from Snowflake and Save")
            st.caption("Pull data from Snowflake, preview, and store it in your local DB.")

            snowflake_sources = {
                "Keppler_Split_Perc": fetch_keppler_split_perc,
                "Vendor_CBM": fetch_vendor_cbm
            }
            source_choice = st.selectbox(
                "Select Snowflake dataset:",
                list(snowflake_sources.keys()),
                index=0,
                key="sf_source_select"
            )

            # Initialize session state for Snowflake fetch
            st.session_state.setdefault("sf_df", None)
            st.session_state.setdefault("sf_source", None)

            # Fetch button
            if st.button("🔄 Fetch from Snowflake", type="primary", key="btn_fetch_sf"):
                df_sf = snowflake_sources[source_choice]()
                if df_sf is not None and not df_sf.empty:
                    st.session_state["sf_df"] = df_sf
                    st.session_state["sf_source"] = source_choice
                    st.success(f"Fetched {len(df_sf):,} rows from {source_choice}.")
                else:
                    st.session_state["sf_df"] = None
                    st.session_state["sf_source"] = None
                    st.info("No data returned.")

            # If we have data, show preview + a separate Save button
            if st.session_state["sf_df"] is not None:
                df_sf = st.session_state["sf_df"]
                src = st.session_state.get("sf_source") or source_choice

                st.write(f"**Preview — {src}** ({len(df_sf):,} rows × {len(df_sf.columns):,} cols)")
                st.dataframe(df_sf.head(200), use_container_width=True, height=460)

                # A persistent message placeholder for save results
                save_msg = st.empty()

                if st.button(f"💾 Save into '{src}'", key="btn_save_sf", help="Save exactly this fetched data into local DB"):
                    try:
                        ok, count = sql_lite_store.save_table(df_sf, src)
                        if ok:
                            save_msg.success(f"Saved {count} row(s) into '{src}'.")
                            # Optionally clear after save:
                            # st.session_state["sf_df"] = None
                        else:
                            save_msg.error("Save failed (save_table returned False).")
                    except Exception as e:
                        save_msg.exception(e)
            else:
                st.info("Click **Fetch from Snowflake** to load data.")

        # ===== Input Sub-tab 3: Review DB =====
        with input_tab3:
            st.subheader("Browse & Export Local DB Tables")
            st.caption("Pick any table from the SQLite file, review its contents, and export to Excel.")

            tables = list_db_tables()
            if not tables:
                st.info("No tables found yet. Load data in the other Input tabs to create tables.")
            else:
                # Refresh before selection to avoid rerun loop on every click
                if st.button("🔄 Refresh list", key="btn_refresh_tables"):
                    st.rerun()

                table_choice = st.selectbox("Select a table to view:", tables, index=0, key="view_table_select")

                # Load selected table
                try:
                    df_tbl = sql_lite_store.load_table(table_choice)
                except Exception as e:
                    st.error(f"Failed to load table '{table_choice}': {e}")
                    df_tbl = pd.DataFrame()

                if not df_tbl.empty:
                    st.success(f"{len(df_tbl):,} row(s) × {len(df_tbl.columns):,} col(s) from '{table_choice}'")
                    st.dataframe(df_tbl, use_container_width=True, height=520)

                    # Export buttons
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "📥 Export to CSV",
                            df_tbl.to_csv(index=False),
                            file_name=f"{table_choice}.csv",
                            mime="text/csv",
                            key="btn_export_csv"
                        )
                    with c2:
                        xlsx_bytes = export_df_to_xlsx_bytes(df_tbl, sheet_name=table_choice[:31] or "Sheet1")
                        st.download_button(
                            "📥 Export to XLSX",
                            xlsx_bytes,
                            file_name=f"{table_choice}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="btn_export_xlsx"
                        )
                else:
                    st.info("Table is empty.")

    # ----------------------- DQ: Sub-tabs -----------------------
    with tab_dq:
        st.subheader("Run DQ Checks & Review Results")

        run_col, _ = st.columns([1, 3])
        with run_col:
            if st.button("🚦 Run DQ Workflow", type="primary", key="btn_run_dq"):
                try:
                    wf.process_DQ_workflow()
                    st.success("DQ workflow completed. Results saved to the local DB.")
                except Exception as e:
                    st.exception(e)

        st.divider()
        st.caption("View DQ artifacts saved to SQLite")
        dq_tables = [
            "DQ_skus_w_no_record_in_plm",
            "DQ_excluded_skus_w_incomplete_data",
            "DQ_skus_w_no_splits",
        ]
        sel = st.selectbox("Pick a DQ table:", dq_tables, index=0, key="dq_view_select")

        # Load + render the chosen table
        try:
            df_view = sql_lite_store.load_table(sel)
            if df_view is None or df_view.empty:
                st.info(f"'{sel}' is empty.")
            else:
                st.success(f"{len(df_view):,} rows × {len(df_view.columns):,} cols in '{sel}'")
                st.dataframe(df_view, use_container_width=True, height=520)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "📥 Export CSV",
                        df_view.to_csv(index=False),
                        file_name=f"{sel}.csv",
                        mime="text/csv",
                        key="btn_export_dq_csv"
                    )
                with c2:
                    xbytes = export_df_to_xlsx_bytes(df_view, sheet_name=sel[:31] or "Sheet1")
                    st.download_button(
                        "📥 Export XLSX",
                        xbytes,
                        file_name=f"{sel}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="btn_export_dq_xlsx"
                    )
        except Exception as e:
            st.exception(e)


    # ----------------------- OUTPUT: Sub-tabs -----------------------
    with tab_output:
        st.subheader("Generate & View Container Plan")

        def list_output_tables():
        # 1) Try a native helper if your store exposes one
            try:
                names = sql_lite_store.list_tables()
                if names:
                    return sorted([t for t in names if str(t).startswith("output_df_")])
            except Exception:
                pass


        run_col, _ = st.columns([1, 3])
        with run_col:
            if st.button("📦 Build Container Plan", type="primary", key="btn_run_container_plan"):
                try:
                    wf.process_containerPlan_workflow()
                    st.success("Container plan built and saved to the local DB.")
                except Exception as e:
                    st.exception(e)

        st.divider()
        st.caption("View output tables (prefix: 'output_')")

        # Discover tables and let user choose
        output_tables = list_output_tables()
        if not output_tables:
            st.info("No tables found with prefix 'output_'. Click **Build Container Plan** or check your DB.")
        else:
            sel = st.selectbox(
                "Pick an output table:",
                output_tables,
                index=0,
                key="output_table_select"
            )

            # Load + render the chosen table
            try:
                df_plan = sql_lite_store.load_table(sel)
                if df_plan is None or df_plan.empty:
                    st.info(f"'{sel}' is empty.")
                else:
                    st.success(f"{len(df_plan):,} rows × {len(df_plan.columns):,} cols in '{sel}'")
                    st.dataframe(df_plan, use_container_width=True, height=520)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "📥 Export CSV",
                            df_plan.to_csv(index=False),
                            file_name=f"{sel}.csv",
                            mime="text/csv",
                            key=f"btn_export_{sel}_csv"
                        )
                    with c2:
                        xbytes = export_df_to_xlsx_bytes(df_plan, sheet_name=sel[:31] or "Sheet1")
                        st.download_button(
                            "📥 Export XLSX",
                            xbytes,
                            file_name=f"{sel}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"btn_export_{sel}_xlsx"
                        )
            except Exception as e:
                st.exception(e)



if __name__ == "__main__":
    main()
