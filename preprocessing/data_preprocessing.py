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

    # Normalize column names a bit
    df_m.columns = [c.strip() for c in df_m.columns]
    if "product_part_number" not in df_m.columns:
        raise KeyError("df_margin must contain column 'product_part_number'")

    # Rename for merge alignment
    df_m = df_m.rename(columns={"product_part_number": "SKU"})

    # Convert SKU to consistent type
    df_sku["SKU"] = df_sku["SKU"].astype(str)
    df_m["SKU"] = df_m["SKU"].astype(str)

    # Select only the columns you care about from the margin table
    margin_cols_to_keep = ["SKU", "product_margin_per_unit"]
    # (you could append 'product_margin', 'cogs', etc. here if you want)
    margin_cols_to_keep = [c for c in margin_cols_to_keep if c in df_m.columns]
    df_m_small = df_m[margin_cols_to_keep]

    # Left join so you keep all SKUs from df_sku_data
    df_merged = df_sku.merge(df_m_small, on="SKU", how="left")

    return df_merged

def enrich_sku_data_with_projection(
    df_sku_data: pd.DataFrame,
    df_fcst: pd.DataFrame,
    *,
    today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Enrich df_sku_data with:
      - DW_FCST: total forecast over df_fcst horizon for the SKU
      - demand_within_LT: sum of forecast demand within LT horizon
      - projected_OH_end_LT: OH + OO - demand_within_LT
      - avg_4wk_runrate: backward-looking 4-week (28-day) avg runrate before LT end
      - DOS_end_LT_days: projected_OH_end_LT / avg_4wk_runrate

      - projected_OH_end_LT_plus4w: projected OH at end of (LT + 4 weeks)
      - DOS_end_LT_plus4w_days: projected_OH_end_LT_plus4w / avg_4wk_runrate

    Inputs:
      df_sku_data.columns (at minimum):
        ['SKU', 'OH', 'AVG_LT', 'OO', ...]
      df_fcst: result of SQL_SKU_FCST
        columns: ['SKU', 'FORECAST_DATE', 'DW_FCST']
    """

    # --- Work on copies to avoid mutating caller's frames in-place ---
    df_supply = df_sku_data.copy()
    df_fcst = df_fcst.copy()

    # Normalize column names for safety
    df_supply.columns = [c.strip().upper() for c in df_supply.columns]
    df_fcst.columns = [c.strip().upper() for c in df_fcst.columns]

    # Required columns in supply snapshot
    req_supply = ["SKU", "OH", "AVG_LT", "OO"]
    for col in req_supply:
        if col not in df_supply.columns:
            raise KeyError(f"df_sku_data is missing required column: {col}")

    # Required columns in forecast
    req_fcst = ["SKU", "FORECAST_DATE", "DW_FCST"]
    for col in req_fcst:
        if col not in df_fcst.columns:
            raise KeyError(f"df_fcst is missing required column: {col}")

    # Dtypes
    df_supply["OH"] = pd.to_numeric(df_supply["OH"], errors="coerce").fillna(0.0)
    df_supply["AVG_LT"] = pd.to_numeric(df_supply["AVG_LT"], errors="coerce").fillna(0.0)
    df_supply["OO"] = pd.to_numeric(df_supply["OO"], errors="coerce").fillna(0.0)

    df_fcst["FORECAST_DATE"] = pd.to_datetime(df_fcst["FORECAST_DATE"])
    df_fcst["DW_FCST"] = pd.to_numeric(df_fcst["DW_FCST"], errors="coerce").fillna(0.0)

    # "Today" anchor for horizons
    if today is None:
        today = pd.Timestamp("today").normalize()

    # Pre-calc: day offsets from today and sort
    df_fcst = df_fcst.sort_values(["SKU", "FORECAST_DATE"])
    df_fcst["DAY_OFFSET"] = (df_fcst["FORECAST_DATE"] - today).dt.days

    # Keep only non-negative offsets (>= today)
    df_fcst = df_fcst[df_fcst["DAY_OFFSET"] >= 0].copy()

    # Group forecasts by SKU
    grouped_fcst = df_fcst.groupby("SKU")

    # Precompute total DW_FCST over full available horizon per SKU
    total_fcst_by_sku = (
        df_fcst.groupby("SKU")["DW_FCST"]
        .sum()
        .rename("DW_FCST_TOTAL")
    )

    # Prepare result columns (aligned with df_supply index)
    demand_within_LT = []
    projected_OH_end_LT = []
    avg_4wk_runrate = []
    dos_end_LT_days = []

    # new: LT+4w
    projected_OH_end_LT_plus4w = []
    dos_end_LT_plus4w_days = []

    for idx, srow in df_supply.iterrows():
        sku = str(srow["SKU"])
        oh = float(srow["OH"])
        oo = float(srow["OO"])
        lt_days = float(srow["AVG_LT"])

        # Lead time → weeks (ceil) → days horizon
        if lt_days <= 0:
            lt_weeks = 0
            horizon_days = 0
        else:
            lt_weeks = math.ceil(lt_days / 7.0)
            horizon_days = lt_weeks * 7

        # For the "LT + 4 weeks" scenario, add 28 days
        if horizon_days > 0:
            horizon_days_plus4w = horizon_days + 28
        else:
            horizon_days_plus4w = 0

        # This SKU's forecast time series (indexed by DAY_OFFSET)
        if sku in grouped_fcst.groups:
            fsku = grouped_fcst.get_group(sku)
        else:
            fsku = pd.DataFrame(columns=["DAY_OFFSET", "DW_FCST"])

        # Build up to max_needed index with zeros
        # Need to cover up to LT+4w horizon
        if horizon_days_plus4w > 0:
            max_needed = max(horizon_days_plus4w - 1, 0)
        else:
            max_needed = 0

        if not fsku.empty:
            ts = (
                fsku.set_index("DAY_OFFSET")["DW_FCST"]
                .reindex(range(0, max_needed + 1), fill_value=0.0)
            )
        else:
            ts = pd.Series(0.0, index=range(0, max_needed + 1), dtype=float)

        # 1) Demand over LT horizon: [0 .. horizon_days-1]
        if horizon_days > 0:
            demand_lt_val = float(ts.loc[0 : horizon_days - 1].sum())
        else:
            demand_lt_val = 0.0

        # 1b) Demand over LT + 4 weeks: [0 .. horizon_days_plus4w - 1]
        if horizon_days_plus4w > 0:
            demand_lt_plus4w_val = float(ts.loc[0 : horizon_days_plus4w - 1].sum())
        else:
            demand_lt_plus4w_val = 0.0

        # 2) Projected OH at end of LT (all OO assumed to arrive within LT)
        proj_oh_val = oh + oo - demand_lt_val

        # 2b) Projected OH at end of LT + 4 weeks
        proj_oh_plus4w_val = oh + oo - demand_lt_plus4w_val

        # 3) Backward-looking 4-week runrate prior to end of LT
        if horizon_days <= 0:
            avg_runrate_4w_val = np.nan
        else:
            window_days = 28
            window_end = horizon_days - 1
            window_start = max(0, window_end - window_days + 1)
            window_len = window_end - window_start + 1

            if window_len <= 0:
                avg_runrate_4w_val = np.nan
            else:
                idx_needed = range(window_start, window_end + 1)
                ts_window = ts.reindex(idx_needed, fill_value=0.0)
                avg_runrate_4w_val = float(ts_window.sum() / window_len)

        # 4) DOS at end of LT
        if avg_runrate_4w_val and avg_runrate_4w_val > 0:
            dos_val = proj_oh_val / avg_runrate_4w_val
            dos_plus4w_val = proj_oh_plus4w_val / avg_runrate_4w_val
        else:
            dos_val = np.nan
            dos_plus4w_val = np.nan

        demand_within_LT.append(demand_lt_val)
        projected_OH_end_LT.append(proj_oh_val)
        avg_4wk_runrate.append(avg_runrate_4w_val)
        dos_end_LT_days.append(dos_val)

        projected_OH_end_LT_plus4w.append(proj_oh_plus4w_val)
        dos_end_LT_plus4w_days.append(dos_plus4w_val)

    # Attach new columns back to df_supply
    df_supply["DEMAND_WITHIN_LT"] = demand_within_LT
    df_supply["PROJECTED_OH_END_LT"] = projected_OH_end_LT
    df_supply["AVG_4WK_RUNRATE"] = avg_4wk_runrate
    df_supply["DOS_END_LT_DAYS"] = dos_end_LT_days

    df_supply["PROJECTED_OH_END_LT_PLUS4W"] = projected_OH_end_LT_plus4w
    df_supply["DOS_END_LT_PLUS4W_DAYS"] = dos_end_LT_plus4w_days

    # Attach total DW_FCST over the horizon as requested DW_FCST
    df_supply = df_supply.merge(
        total_fcst_by_sku,
        on="SKU",
        how="left",
    )

    # Rename DW_FCST_TOTAL to DW_FCST for your requested column name
    df_supply = df_supply.rename(columns={"DW_FCST_TOTAL": "DW_FCST"})

    # Restore original column name casing to match df_sku_data as much as possible
    rename_back = {
        "CHW_SKU_NUMBER": "CHW_SKU_NUMBER",
        "PLANNED_DEMAND": "Planned_Demand",
        "MC1_NAME": "MC1_NAME",
        "MC2_NAME": "MC2_NAME",
        "MC3_NAME": "MC3_NAME",
        "BRAND": "BRAND",
        "CHW_MOQ_LEVEL": "CHW_MOQ_LEVEL",
        "CHW_PRIMARY_SUPPLIER_NAME": "CHW_PRIMARY_SUPPLIER_NAME",
        "CHW_PRIMARY_SUPPLIER_NUMBER": "CHW_PRIMARY_SUPPLIER_NUMBER",
        "CHW_OTB": "CHW_OTB",
        "CHW_MASTER_CASE_PACK": "CHW_MASTER_CASE_PACK",
        "CHW_MASTER_CARTON_CBM": "CHW_MASTER_CARTON_CBM",
        "SKU": "SKU",
        "PRODUCT_NAME": "PRODUCT_NAME",
        "OH": "OH",
        "T90_DAILY_AVG": "T90_DAILY_AVG",
        "F90_DAILY_AVG": "F90_DAILY_AVG",
        "AVG_LT": "AVG_LT",
        "OO": "OO",
        "NEXT_DELIVERY": "NEXT_DELIVERY",
        "T90_DOS_OH": "T90_DOS_OH",
        "F90_DOS_OH": "F90_DOS_OH",
        "F90_DOS_OO": "F90_DOS_OO",
        "T90_BELOW": "T90_BELOW",
        "F90_BELOW": "F90_BELOW",
        "ALERT": "ALERT",
        # new columns
        "DW_FCST": "DW_FCST",
        "DEMAND_WITHIN_LT": "demand_within_LT",
        "PROJECTED_OH_END_LT": "projected_OH_end_LT",
        "AVG_4WK_RUNRATE": "avg_4wk_runrate",
        "DOS_END_LT_DAYS": "DOS_end_LT_days",
        "PROJECTED_OH_END_LT_PLUS4W": "projected_OH_end_LT_plus4w",
        "DOS_END_LT_PLUS4W_DAYS": "DOS_end_LT_plus4w_days",
    }

    rename_back = {k: v for k, v in rename_back.items() if k in df_supply.columns}
    df_supply = df_supply.rename(columns=rename_back)

    return df_supply

def keep_first_max_avglt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AVG_LT"] = pd.to_numeric(df["AVG_LT"], errors="coerce")
    idx = df.groupby("SKU")["AVG_LT"].idxmax()   # index of max AVG_LT per SKU
    return df.loc[idx].reset_index(drop=True)



def process_demand_data() -> pd.DataFrame:

    df_demand = sql_lite_store.load_table("demand_data")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")

    config = get_snowflake_config()
    conn = setconnection(config)
    df_kepplerSplits = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)
    df_sku_fcst = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_SKU_FCST)
    df_margin = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_SKU_MARGIN)

    #saving for reference
    sql_lite_store.save_table(df_kepplerSplits, "Keppler_Split_Perc") 
    sql_lite_store.save_table(df_sku_fcst, "SKU_FCST") 
    sql_lite_store.save_table(df_margin, "SKU_MARGIN") 


    df_skuSupplySnapshot = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_SKU_Supply_Snapshot)
    df_vendor_cbm = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_Vendor_CBM)

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
    'T90_DAILY_AVG', 'F90_DAILY_AVG', 'AVG_LT', 'OO', 'NEXT_DELIVERY',
    'T90_DOS_OH', 'F90_DOS_OH', 'F90_DOS_OO', 'T90_BELOW', 'F90_BELOW',
    'ALERT',]
    
    df_skuSupplySnapshot = df_skuSupplySnapshot[keep_cols_2].copy()

    df_demand = df_demand[["product_part_number", "Final Buy Qty"]].copy()
    df_demand = df_demand.rename(columns={'product_part_number': 'CHW_SKU_NUMBER', 'Final Buy Qty': 'Planned_Demand'})
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
    num_cols = ["AVG_LT", "F90_DAILY_AVG", "OH", "OO", "Planned_Demand", "CHW_MOQ_LEVEL"]
    df_sku_data[num_cols] = df_sku_data[num_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    df_sku_data['AVG_LT'] = df_sku_data['AVG_LT'].fillna(avg_lt_mean)
    df_sku_data['OO'] = df_sku_data['OO'].fillna(0)
    df_sku_data['CHW_MOQ_LEVEL'] = df_sku_data['CHW_MOQ_LEVEL'].fillna(0)
    df_sku_data['OH'] = df_sku_data['OH'].fillna(0)

    df_sku_data = enrich_sku_data_with_projection(df_sku_data,df_sku_fcst)
    df_sku_data = add_margin_to_sku_data(df_sku_data,df_margin)

    df_sku_data["baseConsumption"] = np.where(
        df_sku_data["F180_DAILY_AVG"].notna(),
        (df_sku_data["AVG_LT"] + 4 * 7) * df_sku_data["F180_DAILY_AVG"],
        0.0
    )

    base_qty = df_sku_data["baseConsumption"] - (df_sku_data["OH"].fillna(0) + df_sku_data["OO"].fillna(0))

    df_sku_data["baseDemand"] = np.maximum(base_qty, 0)
    df_sku_data["baseDemand"] = np.minimum(df_sku_data["baseDemand"], df_sku_data["Planned_Demand"].fillna(0))

    df_sku_data["proj_OH"] = df_sku_data["OH"] + df_sku_data["OO"]  - df_sku_data["baseConsumption"]

    #probably not needed. commenting out for now.
    # df_sku_data["baseDemand"] = np.where(
    #     df_sku_data["baseConsumption"] == 0,
    #     df_sku_data["Planned_Demand"],             # if baseConsumption == 0 → use planned_demand
    #     df_sku_data["baseDemand"]                  # otherwise keep existing baseDemand
    # )

    # #snapping baseDemand to MOQ, since its not really a choice
    # df_sku_data["baseDemand"] = np.maximum(df_sku_data["baseDemand"], df_sku_data["CHW_MOQ_LEVEL"])

    #snapping baseDemand to the higher of mcp multiples
    m = df_sku_data["CHW_MASTER_CASE_PACK"]
    df_sku_data["baseDemand"] = np.ceil(df_sku_data["baseDemand"] / m) * m
    df_sku_data["excess_demand"] = np.floor(df_sku_data["excess_demand"] / m) * m

    #taking out suppliers that have no demand
    suppliers_with_demand = (
        df_sku_data.loc[df_sku_data["Planned_Demand"].notna(), "CHW_PRIMARY_SUPPLIER_NUMBER"]
        .dropna()
        .unique()
    )

    df_sku_data = df_sku_data[
        df_sku_data["CHW_PRIMARY_SUPPLIER_NUMBER"].isin(suppliers_with_demand)
    ]

    #defaulting columns to 0 if they are null
    cols_to_fill_zero = ["Planned_Demand","CHW_MOQ_LEVEL", "OH", "OO", "T90_DAILY_AVG", "F90_DAILY_AVG", 
    "T90_DOS_OH", "F90_DOS_OH", "F90_DOS_OO", "T90_BELOW", "F90_BELOW", "baseConsumption",
    "bufferConsumption", "baseDemand", "bufferDemand", "excess_demand"]
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
    then split Planned_Demand into TLA1/TNY1/MDT1 using *_FRAC.

    If no match in df_kepplerSplits, assign all demand to TNY1 and 0 to the others.

    Then:
      - Compute per-destination demand (Planned_Demand_dest)
      - Convert to cases using CHW_MASTER_CASE_PACK
      - If Planned_Demand is an exact multiple of MCP for a SKU, ensure that
        the sum of cases across DEST equals Planned_Demand / MCP (no inflation),
        using a largest-remainder allocation.

    Output: long dataframe with DEST, Planned_Demand_dest, and Planned_Demand_cases_need.
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
    long["Planned_Demand"] = pd.to_numeric(long["Planned_Demand"], errors="coerce").fillna(0.0)
    long["Planned_Demand_dest"] = long["Planned_Demand"] * long["frac"]

    # --- prepare for MCP snapping with "no inflation when divisible" logic ---
    long["CHW_MASTER_CASE_PACK"] = pd.to_numeric(long["CHW_MASTER_CASE_PACK"], errors="coerce")
    long["Planned_Demand_dest"] = pd.to_numeric(long["Planned_Demand_dest"], errors="coerce").fillna(0)

    def _allocate_cases_per_sku(group: pd.DataFrame) -> pd.DataFrame:
        """
        For a single SKU (CHW_SKU_NUMBER group):
          - If Planned_Demand is an exact multiple of MCP, keep total cases fixed.
          - Otherwise, ceil per-dest cases independently.
        """
        # Take MCP and total demand for this SKU (assumed constant within group)
        mcp = group["CHW_MASTER_CASE_PACK"].iloc[0]
        total_planned = group["Planned_Demand"].iloc[0]

        # Fallback: no MCP or no demand -> simple ceil per destination
        if pd.isna(mcp) or mcp <= 0 or total_planned <= 0:
            safe_mcp = mcp if (mcp and mcp > 0) else np.nan
            raw_cases = group["Planned_Demand_dest"] / safe_mcp
            cases = np.ceil(raw_cases).fillna(0).clip(lower=0).astype("Int64")
            group["Planned_Demand_cases_need"] = cases
            return group

        # Target total cases from original total demand
        target_total_cases = total_planned / mcp

        # If total_planned not an exact multiple of MCP -> simple ceil per dest
        if not np.isclose(target_total_cases, round(target_total_cases)):
            raw_cases = group["Planned_Demand_dest"] / mcp
            cases = np.ceil(raw_cases).fillna(0).clip(lower=0).astype("Int64")
            group["Planned_Demand_cases_need"] = cases
            return group

        # Here: Planned_Demand is nicely divisible by MCP
        target_total_cases = int(round(target_total_cases))

        raw_cases = group["Planned_Demand_dest"] / mcp
        base_cases = np.floor(raw_cases)
        frac = raw_cases - base_cases

        base_sum = int(base_cases.sum())

        # If floor allocations already hit the target, we're done
        if base_sum == target_total_cases:
            group["Planned_Demand_cases_need"] = base_cases.astype("Int64")
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
        group["Planned_Demand_cases_need"] = cases
        return group

    # Group by SKU – using CHW_SKU_NUMBER as the key
    long = long.groupby("CHW_SKU_NUMBER", group_keys=False).apply(_allocate_cases_per_sku)

    return long



#putting them together
if __name__ == "__main__":

    config = get_snowflake_config()
    conn = setconnection(config)

    df_sku_data = process_demand_data()
    # ok, count = sql_lite_store.save_table(df_sku_data, "df_sku_data")
    # print(f"Saved {count} rows to df_sku_data")
    df_sku_data = sql_lite_store.load_table("df_sku_data")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")
    df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
    #df_kepplerSplits = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)

    demand_by_Dest = split_base_demand_by_dest(df_sku_data, df_kepplerSplits)

    #generating state objects for langgraph workflow
    sku_data_state_list = state_loader.df_to_chewy_sku_states(df_sku_data)
    container_plan_rows = state_loader.load_container_plan_rows(demand_by_Dest)

    vendor_state_list = state_loader.df_to_vendor_states(df_sku_data, df_CBM_Max, sku_data_state_list, container_plan_rows)

    print(vendor_state_list)


    
