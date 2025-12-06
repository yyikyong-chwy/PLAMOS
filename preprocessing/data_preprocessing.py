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
      - DW_FCST: total forecast across all available future days
      - demand_within_LT: demand within LT (actual days, not weekly-rounded)
      - projected_OH_end_LT: OH + OO - demand_within_LT
      - avg_4wk_runrate: 28-day backward-looking average runrate before LT end
      - DOS_end_LT_days: projected_OH_end_LT / avg_4wk_runrate

      - projected_OH_end_LT_plus4w: inventory at end of LT + 28 days
      - DOS_end_LT_plus4w_days: projected_OH_end_LT_plus4w / avg_4wk_runrate
    """

    # Copy inputs
    df_supply = df_sku_data.copy()
    df_fcst = df_fcst.copy()

    # Normalize column names to uppercase internally
    df_supply.columns = [c.upper() for c in df_supply.columns]
    df_fcst.columns = [c.upper() for c in df_fcst.columns]

    # Required columns
    for col in ["SKU", "OH", "AVG_LT", "OO"]:
        if col not in df_supply.columns:
            raise KeyError(f"Missing required supply column: {col}")

    for col in ["SKU", "FORECAST_DATE", "DW_FCST"]:
        if col not in df_fcst.columns:
            raise KeyError(f"Missing required forecast column: {col}")

    # Convert types
    df_supply["OH"] = pd.to_numeric(df_supply["OH"], errors="coerce").fillna(0)
    df_supply["OO"] = pd.to_numeric(df_supply["OO"], errors="coerce").fillna(0)
    df_supply["AVG_LT"] = pd.to_numeric(df_supply["AVG_LT"], errors="coerce").fillna(0)

    df_fcst["FORECAST_DATE"] = pd.to_datetime(df_fcst["FORECAST_DATE"])
    df_fcst["DW_FCST"] = pd.to_numeric(df_fcst["DW_FCST"], errors="coerce").fillna(0)

    # Anchor date
    if today is None:
        today = pd.Timestamp("today").normalize()

    # Compute day offsets
    df_fcst = df_fcst.sort_values(["SKU", "FORECAST_DATE"])
    df_fcst["DAY_OFFSET"] = (df_fcst["FORECAST_DATE"] - today).dt.days
    df_fcst = df_fcst[df_fcst["DAY_OFFSET"] >= 0]   # Keep today and future

    # Group forecasts by SKU
    grouped_fcst = df_fcst.groupby("SKU")

    # Precompute total FCST across full available horizon
    df_total_fcst = (
        df_fcst.groupby("SKU")["DW_FCST"]
        .sum()
        .rename("DW_FCST")
        .reset_index()
    )

    # Output arrays
    demand_LT = []
    proj_OH_LT = []
    avg_runrate_4w = []
    dos_LT = []
    proj_OH_LT_plus4w = []
    dos_LT_plus4w = []

    for _, row in df_supply.iterrows():
        sku = str(row["SKU"])
        oh = row["OH"]
        oo = row["OO"]
        lt_days = max(0, math.ceil(row["AVG_LT"]))  # LT in real days

        # LT + 4 weeks
        lt_plus4w = lt_days + 28

        # Build time-series for this SKU
        if sku in grouped_fcst.groups:
            sub = grouped_fcst.get_group(sku).set_index("DAY_OFFSET")["DW_FCST"]
        else:
            sub = pd.Series(dtype=float)

        # Ensure TS covers LT + 4w horizon
        max_needed = max(lt_plus4w - 1, 0)
        ts = sub.reindex(range(0, max_needed + 1), fill_value=0.0)

        # Demand during LT
        d_LT = ts.loc[0:lt_days - 1].sum() if lt_days > 0 else 0.0

        # Demand during LT + 4w
        d_LT_plus4 = ts.loc[0:lt_plus4w - 1].sum() if lt_plus4w > 0 else 0.0

        # Projected OH at end of LT
        oh_LT = oh + oo - d_LT

        # Projected OH at end of LT + 4w
        oh_LT_plus4 = oh + oo - d_LT_plus4

        # Backward 4-week runrate before LT end
        if lt_days > 0:
            w_end = lt_days - 1
            w_start = max(0, w_end - 27)
            window = ts.loc[w_start:w_end]
            runrate_4w = window.mean() if len(window) > 0 else np.nan
        else:
            runrate_4w = np.nan

        # DOS at LT and LT+4w
        dos_val = oh_LT / runrate_4w if runrate_4w and runrate_4w > 0 else np.nan
        dos_val_plus4 = oh_LT_plus4 / runrate_4w if runrate_4w and runrate_4w > 0 else np.nan

        # Append
        demand_LT.append(d_LT)
        proj_OH_LT.append(oh_LT)
        avg_runrate_4w.append(runrate_4w)
        dos_LT.append(dos_val)
        proj_OH_LT_plus4w.append(oh_LT_plus4)
        dos_LT_plus4w.append(dos_val_plus4)

    # Assign new columns
    df_supply["demand_within_LT"] = demand_LT
    df_supply["projected_OH_end_LT"] = proj_OH_LT
    df_supply["avg_4wk_runrate"] = avg_runrate_4w
    df_supply["DOS_end_LT_days"] = dos_LT
    df_supply["projected_OH_end_LT_plus4w"] = proj_OH_LT_plus4w
    df_supply["DOS_end_LT_plus4w_days"] = dos_LT_plus4w

    # Add DW_FCST total
    df_supply = df_supply.merge(df_total_fcst, on="SKU", how="left")

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
    'T90_DOS_OH', 'F90_DOS_OH', 'F90_DOS_OO', 'F180_DOS_OH', 'F180_DOS_OO']
    
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
        (df_sku_data[["OO","OH","PLANNED_DEMAND","demand_within_LT","projected_OH_end_LT","avg_4wk_runrate"]] != 0).any(axis=1)]
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


    
