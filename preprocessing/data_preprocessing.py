import pandas as pd
import sqlite3
import sys
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from data.snowflake_pull import get_snowflake_config, setconnection, run_query_to_df, mutate_keppler_splits
import data.sql_lite_store as sql_lite_store
import data.snowflake_pull as snowflake_pull


import states.state_loader as state_loader


import pandas as pd
import numpy as np
import math

import pandas as pd
import numpy as np
import math

def add_margin_to_sku_data(
    df_sku_data: pd.DataFrame,
    df_margin: pd.DataFrame) -> pd.DataFrame:
    """
    Add product_margin_per_unit (and optionally other margin fields)
    from df_margin (SQL_SKU_MARGIN output) to df_sku_data.

    df_sku_data: must contain column 'SKU'
    df_margin: must contain columns
        - 'product_part_number'
        - 'product_margin_per_unit'
        (plus all the other margin metrics if you want them)
    """

    # Work on copies
    df_sku = df_sku_data.copy()
    df_m = df_margin.copy()

    # Select only the columns you care about from the margin table
    margin_cols_to_keep = ["CHW_SKU_NUMBER", "PRODUCT_MARGIN_PER_UNIT"]
    # (you could append 'product_margin', 'cogs', etc. here if you want)
    margin_cols_to_keep = [c for c in margin_cols_to_keep if c in df_m.columns]
    df_m_small = df_m[margin_cols_to_keep]

    # Left join so you keep all SKUs from df_sku_data
    df_merged = df_sku.merge(df_m_small, on="CHW_SKU_NUMBER", how="left")

    return df_merged


def enrich_sku_data_with_projection(
    df_sku_data: pd.DataFrame,
    df_fcst: pd.DataFrame,
    *,
    today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Enrich df_sku_data with daily-forecast–driven LT projections.

    Added Columns:
      - consumption_within_LT: consumption within LT (actual days)
      - consumption_within_LT_plus4w: consumption within LT + 4 weeks
      - projected_OH_end_LT: OH + OO - consumption_within_LT
      - runrate_at_LT: 28-day backward-looking average runrate before LT end
      - DOS_end_LT_days: projected_OH_end_LT / runrate_at_LT
      - projected_OH_end_LT_plus4w: inventory at end of LT + 28 days
      - runrate_at_LT_plus4w: 28-day backward-looking average runrate before LT+4w end
      - DOS_end_LT_plus4w_days: projected_OH_end_LT_plus4w / runrate_at_LT_plus4w

    All columns from df_sku_data are preserved.
    """
    df_supply = df_sku_data.copy()
    df_fcst = df_fcst.copy()

    # Normalize column names to uppercase
    df_supply.columns = [c.upper() for c in df_supply.columns]
    df_fcst.columns = [c.upper() for c in df_fcst.columns]

    # Validate required columns
    required_supply = ["CHW_SKU_NUMBER", "OH", "AVG_LT", "OO"]
    required_fcst = ["SKU", "FORECAST_DATE", "DW_FCST"]
    
    for col in required_supply:
        if col not in df_supply.columns:
            raise KeyError(f"Missing required supply column: {col}")
    for col in required_fcst:
        if col not in df_fcst.columns:
            raise KeyError(f"Missing required forecast column: {col}")

    # Convert types for supply data
    df_supply["OH"] = pd.to_numeric(df_supply["OH"], errors="coerce").fillna(0)
    df_supply["OO"] = pd.to_numeric(df_supply["OO"], errors="coerce").fillna(0)
    df_supply["AVG_LT"] = pd.to_numeric(df_supply["AVG_LT"], errors="coerce").fillna(0)

    # Convert types for forecast data
    df_fcst["FORECAST_DATE"] = pd.to_datetime(df_fcst["FORECAST_DATE"])
    df_fcst["DW_FCST"] = pd.to_numeric(df_fcst["DW_FCST"], errors="coerce").fillna(0)

    # Anchor date
    if today is None:
        today = pd.Timestamp("today").normalize()

    # Trim forecast to today onwards and sort by SKU/date
    df_fcst = df_fcst[df_fcst["FORECAST_DATE"] >= today].copy()
    df_fcst = df_fcst.sort_values(["SKU", "FORECAST_DATE"]).reset_index(drop=True)

    # Build lookup: SKU -> list of daily forecasts (ordered by date from today)
    fcst_by_sku = {
        sku: grp["DW_FCST"].tolist()
        for sku, grp in df_fcst.groupby("SKU")
    }

    # Pre-allocate result arrays
    n_rows = len(df_supply)
    results = {
        "consumption_within_LT": np.zeros(n_rows),
        "consumption_within_LT_plus4w": np.zeros(n_rows),
        "projected_OH_end_LT": np.zeros(n_rows),
        "runrate_at_LT": np.full(n_rows, np.nan),
        "DOS_end_LT_days": np.full(n_rows, np.nan),
        "projected_OH_end_LT_plus4w": np.zeros(n_rows),
        "runrate_at_LT_plus4w": np.full(n_rows, np.nan),
        "DOS_end_LT_plus4w_days": np.full(n_rows, np.nan),
    }

    for i, row in enumerate(df_supply.itertuples(index=False)):
        sku = str(row.CHW_SKU_NUMBER)
        oh = row.OH
        oo = row.OO
        lt_days = max(0, math.ceil(row.AVG_LT))
        lt_plus4w = lt_days + 28

        # Get forecast list for this SKU (empty if not found)
        fcst_list = fcst_by_sku.get(sku, [])

        # Pad with zeros if forecast is shorter than needed horizon
        max_needed = lt_plus4w
        if len(fcst_list) < max_needed:
            fcst_list = fcst_list + [0.0] * (max_needed - len(fcst_list))

        # Consumption within LT (days 0 to lt_days-1)
        c_LT = sum(fcst_list[:lt_days]) if lt_days > 0 else 0.0

        # Consumption within LT + 4w (days 0 to lt_plus4w-1)
        c_LT_plus4w = sum(fcst_list[:lt_plus4w]) if lt_plus4w > 0 else 0.0

        # Projected OH at end of LT and LT+4w
        oh_LT = oh + oo - c_LT
        oh_LT_plus4w = oh + oo - c_LT_plus4w

        # Backward 28-day runrate ending at LT
        if lt_days > 0:
            w_start = max(0, lt_days - 28)
            window = fcst_list[w_start:lt_days]
            runrate_LT = np.mean(window) if window else np.nan
        else:
            runrate_LT = np.nan

        # Backward 28-day runrate ending at LT+4w
        if lt_plus4w > 0:
            w_start_plus4w = max(0, lt_plus4w - 28)
            window_plus4w = fcst_list[w_start_plus4w:lt_plus4w]
            runrate_LT_plus4w = np.mean(window_plus4w) if window_plus4w else np.nan
        else:
            runrate_LT_plus4w = np.nan

        # DOS calculations
        dos_LT = oh_LT / runrate_LT if runrate_LT and runrate_LT > 0 else np.nan
        dos_LT_plus4w = oh_LT_plus4w / runrate_LT_plus4w if runrate_LT_plus4w and runrate_LT_plus4w > 0 else np.nan

        # Store results
        results["consumption_within_LT"][i] = c_LT
        results["consumption_within_LT_plus4w"][i] = c_LT_plus4w
        results["projected_OH_end_LT"][i] = oh_LT
        results["runrate_at_LT"][i] = runrate_LT
        results["DOS_end_LT_days"][i] = dos_LT
        results["projected_OH_end_LT_plus4w"][i] = oh_LT_plus4w
        results["runrate_at_LT_plus4w"][i] = runrate_LT_plus4w
        results["DOS_end_LT_plus4w_days"][i] = dos_LT_plus4w

    # Assign all new columns at once
    for col_name, values in results.items():
        df_supply[col_name] = values

    return df_supply

def keep_first_max_avglt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AVG_LT"] = pd.to_numeric(df["AVG_LT"], errors="coerce")
    idx = df.groupby("SKU")["AVG_LT"].idxmax()   # index of max AVG_LT per SKU
    return df.loc[idx].reset_index(drop=True)



def process_demand_data(LOAD_FROM_SQL_LITE: bool = False) -> pd.DataFrame:

    df_demand = sql_lite_store.load_table("demand_data")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")

    if not LOAD_FROM_SQL_LITE:
        config = get_snowflake_config()
        conn = setconnection(config)
        df_kepplerSplits = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)
        df_sku_fcst = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_SKU_FCST)
        df_margin = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_SKU_MARGIN)
        df_skuSupplySnapshot = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_SKU_Supply_Snapshot)
        df_vendor_cbm = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_Vendor_CBM)

        #saving for reference
        sql_lite_store.save_table(df_kepplerSplits, "Keppler_Split_Perc") 
        sql_lite_store.save_table(df_sku_fcst, "SKU_FCST") 
        sql_lite_store.save_table(df_margin, "SKU_MARGIN") 
        sql_lite_store.save_table(df_skuSupplySnapshot, "SKU_Supply_Snapshot") 
        sql_lite_store.save_table(df_vendor_cbm, "Vendor_CBM") 

    else:
        df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
        df_sku_fcst = sql_lite_store.load_table("SKU_FCST")
        df_margin = sql_lite_store.load_table("SKU_MARGIN")
        df_skuSupplySnapshot = sql_lite_store.load_table("SKU_Supply_Snapshot")
        df_vendor_cbm = sql_lite_store.load_table("Vendor_CBM")

    keep_cols_1 = [
        'MC1_NAME', 'MC2_NAME', 'MC3_NAME', 'BRAND',
        "CHW_SKU_NUMBER",
        "CHW_MOQ_LEVEL",
        "CHW_PRIMARY_SUPPLIER_NAME",
        "CHW_PRIMARY_SUPPLIER_NUMBER",
        'CHW_OTB',
        "CHW_MASTER_CASE_PACK",
        "CHW_MASTER_CARTON_CBM",]
        
    df_vendor_cbm = df_vendor_cbm[keep_cols_1]

    keep_cols_2 = [
    'SKU','PRODUCT_NAME', 'OH',
    'T90_DAILY_AVG', 'F90_DAILY_AVG', 'F180_DAILY_AVG', 'AVG_LT', 'OO',
    'T90_DOS_OH', 'F90_DOS_OH', 'F90_DOS_OO', 'F180_DOS_OH', 'F180_DOS_OO',
    'PRODUCT_ABC_CODE']
    
    df_skuSupplySnapshot = df_skuSupplySnapshot[keep_cols_2].copy()

    df_demand = df_demand[["product_part_number", "Final Buy Qty"]].copy()
    df_demand = df_demand.rename(columns={'product_part_number': 'CHW_SKU_NUMBER', 'Final Buy Qty': 'PLANNED_DEMAND'})
    df_demand['CHW_SKU_NUMBER'] = df_demand['CHW_SKU_NUMBER'].astype(str).str.strip()
    # --- NEW: SKUs in demand but not in vendor_cbm (anti-join) ---
    df_vendor_cbm_keys = df_vendor_cbm[['CHW_SKU_NUMBER']].copy()
    df_vendor_cbm_keys['CHW_SKU_NUMBER'] = df_vendor_cbm_keys['CHW_SKU_NUMBER'].astype(str).str.strip()

    df_demand_vs_vendor_cbm = df_demand.merge(
        df_vendor_cbm_keys,
        on="CHW_SKU_NUMBER",
        how="left",
        indicator=True,
    )

    df_missing_in_vendor_cbm = (
        df_demand_vs_vendor_cbm[df_demand_vs_vendor_cbm["_merge"] == "left_only"]
        .drop(columns=["_merge"])
    )

    # if you want to persist this as a DQ table:
    sql_lite_store.save_table(df_missing_in_vendor_cbm, "DQ_skus_w_no_record_in_plm")

    df_demand = df_demand.merge(
        df_vendor_cbm,
        how='outer',
        on=['CHW_SKU_NUMBER'])

    #combine the 2 tables,
    df_demand['CHW_SKU_NUMBER'] = df_demand['CHW_SKU_NUMBER'].astype(str).str.strip()
    df_skuSupplySnapshot['SKU']= df_skuSupplySnapshot['SKU'].astype(str).str.strip()
    #detected duplicated SKU in df_skuSupplySnapshot, keep the first max AVG_LT
    df_skuSupplySnapshot = keep_first_max_avglt(df_skuSupplySnapshot)


    df_sku_data = df_demand.merge(
        df_skuSupplySnapshot,
        how='outer',
        left_on='CHW_SKU_NUMBER',
        right_on='SKU'
    )

    # df_sku_data = df_sku_data.merge(
    #     df_skuSupplySnapshot,
    #     how='left',
    #     left_on='CHW_SKU_NUMBER',
    #     right_on='SKU'
    # )

    # Drop requested columns (only if present)
    avg_lt_mean = df_sku_data['AVG_LT'].mean(skipna=True)
    num_cols = ["AVG_LT", "F90_DAILY_AVG", "OH", "OO", "PLANNED_DEMAND", "CHW_MOQ_LEVEL"]
    df_sku_data[num_cols] = df_sku_data[num_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    df_sku_data['AVG_LT'] = df_sku_data['AVG_LT'].fillna(avg_lt_mean)
    df_sku_data['OO'] = df_sku_data['OO'].fillna(0)
    df_sku_data['CHW_MOQ_LEVEL'] = df_sku_data['CHW_MOQ_LEVEL'].fillna(0)
    df_sku_data['OH'] = df_sku_data['OH'].fillna(0)
    df_sku_data['PLANNED_DEMAND'] = df_sku_data['PLANNED_DEMAND'].fillna(0)

    df_sku_data = enrich_sku_data_with_projection(df_sku_data,df_sku_fcst)
    df_sku_data = add_margin_to_sku_data(df_sku_data,df_margin)

    #take out bunch of irrelevant skus
    df_sku_data = df_sku_data[
        (df_sku_data[["OO","OH","PLANNED_DEMAND","consumption_within_LT","projected_OH_end_LT","runrate_at_LT"]] != 0).any(axis=1)]
    df_sku_data = df_sku_data[df_sku_data["CHW_PRIMARY_SUPPLIER_NAME"].notna()]

    #taking out suppliers that have no demand
    suppliers_with_demand = (
        df_sku_data.loc[df_sku_data["PLANNED_DEMAND"].notna(), "CHW_PRIMARY_SUPPLIER_NUMBER"]
        .dropna()
        .unique()
    )

    df_sku_data = df_sku_data[
        df_sku_data["CHW_PRIMARY_SUPPLIER_NUMBER"].isin(suppliers_with_demand)
    ]

    #defaulting columns to 0 if they are null
    cols_to_fill_zero = ["PLANNED_DEMAND","CHW_MOQ_LEVEL", "OH", "OO", "T90_DAILY_AVG", "F90_DAILY_AVG", 
    "T90_DOS_OH", "F90_DOS_OH", "F90_DOS_OO", "F180_DOS_OH", "F180_DOS_OO"]
    df_sku_data[cols_to_fill_zero] = df_sku_data[cols_to_fill_zero].fillna(0)


    ok, count = sql_lite_store.save_table(df_sku_data, "df_sku_data")
    print(f"Saved {count} rows to df_sku_data")
    return df_sku_data

import numpy as np
import pandas as pd

def split_base_demand_by_dest(df_sku_data: pd.DataFrame,
                              df_kepplerSplits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge df_sku_data with df_kepplerSplits on CHW_SKU_NUMBER == ITEM_ID,
    then split PLANNED_DEMAND into TLA1/TNY1/MDT1 using *_FRAC.

    If no match in df_kepplerSplits, assign all demand to TNY1 and 0 to the others.

    Then:
      - Compute per-destination demand (PLANNED_DEMAND_DEST)
      - Convert to cases using CHW_MASTER_CASE_PACK
      - If PLANNED_DEMAND is an exact multiple of MCP for a SKU, ensure that
        the sum of cases across DEST equals PLANNED_DEMAND / MCP (no inflation),
        using a largest-remainder allocation.

    Output: long dataframe with DEST, PLANNED_DEMAND_DEST, and PLANNED_DEMAND_cases_need.
    """
    # --- keep only needed columns from splits ---
    columns_to_keep = ['ITEM_ID', 'TLA1_FRAC', 'TNY1_FRAC', 'MDT1_FRAC']
    df_splits = df_kepplerSplits[columns_to_keep]

    # --- left join on SKU ---
    merged = df_sku_data.merge(
        df_splits,
        how="left",
        left_on="CHW_SKU_NUMBER",
        right_on="ITEM_ID",
        suffixes=("", "_split")
    )

    # --- default fractions when no match: all to TNY1 ---
    for col in ["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged["TLA1_FRAC"] = merged["TLA1_FRAC"].astype(float)
    merged["TNY1_FRAC"] = merged["TNY1_FRAC"].astype(float)
    merged["MDT1_FRAC"] = merged["MDT1_FRAC"].astype(float)

    no_match = merged["ITEM_ID"].isna()
    merged.loc[no_match, ["TLA1_FRAC", "MDT1_FRAC"]] = 0.0
    merged.loc[no_match, "TNY1_FRAC"] = 1.0

    # Fill any remaining NaNs with 0
    merged[["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]] = (
        merged[["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]].fillna(0.0)
    )

    # --- long form: DEST + frac ---
    long = merged.melt(
        id_vars=merged.columns.difference(["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]),
        value_vars=["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"],
        var_name="DEST",
        value_name="frac"
    )
    long["DEST"] = long["DEST"].str.replace("_FRAC", "", regex=False)

    # --- compute split demand ---
    long["PLANNED_DEMAND"] = pd.to_numeric(long["PLANNED_DEMAND"], errors="coerce").fillna(0.0)
    long["PLANNED_DEMAND_DEST"] = long["PLANNED_DEMAND"] * long["frac"]

    # --- prepare for MCP snapping with "no inflation when divisible" logic ---
    long["CHW_MASTER_CASE_PACK"] = pd.to_numeric(long["CHW_MASTER_CASE_PACK"], errors="coerce")
    long["PLANNED_DEMAND_DEST"] = pd.to_numeric(long["PLANNED_DEMAND_DEST"], errors="coerce").fillna(0)

    def _allocate_cases_per_sku(group: pd.DataFrame) -> pd.DataFrame:
        """
        For a single SKU (CHW_SKU_NUMBER group):
          - If PLANNED_DEMAND is an exact multiple of MCP, keep total cases fixed.
          - Otherwise, ceil per-dest cases independently.
        """
        # Take MCP and total demand for this SKU (assumed constant within group)
        mcp = group["CHW_MASTER_CASE_PACK"].iloc[0]
        total_planned = group["PLANNED_DEMAND"].iloc[0]

        # Fallback: no MCP or no demand -> simple ceil per destination
        if pd.isna(mcp) or mcp <= 0 or total_planned <= 0:
            safe_mcp = mcp if (mcp and mcp > 0) else np.nan
            raw_cases = group["PLANNED_DEMAND_DEST"] / safe_mcp
            cases = np.ceil(raw_cases).fillna(0).clip(lower=0).astype("Int64")
            group["PLANNED_DEMAND_cases_need"] = cases
            return group

        # Target total cases from original total demand
        target_total_cases = total_planned / mcp

        # If total_planned not an exact multiple of MCP -> simple ceil per dest
        if not np.isclose(target_total_cases, round(target_total_cases)):
            raw_cases = group["PLANNED_DEMAND_DEST"] / mcp
            cases = np.ceil(raw_cases).fillna(0).clip(lower=0).astype("Int64")
            group["PLANNED_DEMAND_cases_need"] = cases
            return group

        # Here: PLANNED_DEMAND is nicely divisible by MCP
        target_total_cases = int(round(target_total_cases))

        raw_cases = group["PLANNED_DEMAND_DEST"] / mcp
        base_cases = np.floor(raw_cases)
        frac = raw_cases - base_cases

        base_sum = int(base_cases.sum())

        # If floor allocations already hit the target, we're done
        if base_sum == target_total_cases:
            group["PLANNED_DEMAND_cases_need"] = base_cases.astype("Int64")
            return group

        # Distribute remaining cases to largest remainders
        remaining = target_total_cases - base_sum
        cases = base_cases.copy()

        if remaining > 0:
            # indices sorted by descending fractional part
            order = np.argsort(-frac.to_numpy())
            for idx in order[:remaining]:
                cases.iloc[idx] += 1

        cases = cases.clip(lower=0).astype("Int64")
        group["PLANNED_DEMAND_cases_need"] = cases
        return group

    # Group by SKU – using CHW_SKU_NUMBER as the key
    long = long.groupby("CHW_SKU_NUMBER", group_keys=False).apply(_allocate_cases_per_sku)

    return long


def run_preprocessing(LOAD_FROM_SQL_LITE: bool = False):

    df_sku_data = process_demand_data( LOAD_FROM_SQL_LITE)
    df_sku_data = sql_lite_store.load_table("df_sku_data")

    if not LOAD_FROM_SQL_LITE:
        config = get_snowflake_config()
        conn = setconnection(config)
        df_kepplerSplits = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)
        sql_lite_store.save_table(df_kepplerSplits, "Keppler_Split_Perc") 
    else:
        df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")

    df_CBM_Max = sql_lite_store.load_table("CBM_Max")
    demand_by_Dest = split_base_demand_by_dest(df_sku_data, df_kepplerSplits)

    #generating state objects for langgraph workflow
    sku_data_state_list = state_loader.df_to_chewy_sku_states(df_sku_data)
    container_plan_rows = state_loader.load_container_plan_rows(demand_by_Dest)
    vendor_state_list = state_loader.df_to_vendor_states(df_sku_data, df_CBM_Max, sku_data_state_list, container_plan_rows)

    return vendor_state_list


#putting them together
if __name__ == "__main__":

    LOAD_FROM_SQL_LITE= True
    vendor_state_list = run_preprocessing(LOAD_FROM_SQL_LITE)


    
