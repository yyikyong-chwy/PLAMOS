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

import data.demand_pull as demand_pull
import states.state_loader as state_loader


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
    df_kepplerSplits = snowflake_pull.mutate_keppler_splits(df_kepplerSplits)

    sql_lite_store.save_table(df_kepplerSplits, "Keppler_Split_Perc") #save a copy for reference
    #df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
    


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

    df_sku_data["baseConsumption"] = np.where(
        df_sku_data["F90_DAILY_AVG"].notna(),
        (df_sku_data["AVG_LT"] + 4 * 7) * df_sku_data["F90_DAILY_AVG"],
        0.0
    )

    df_sku_data["bufferConsumption"] = np.where(
        df_sku_data["F90_DAILY_AVG"].notna(),
        (df_sku_data["AVG_LT"] + 8 * 7) * df_sku_data["F90_DAILY_AVG"],
        df_sku_data["Planned_Demand"]
    )

    base_qty = df_sku_data["baseConsumption"] - (df_sku_data["OH"].fillna(0) + df_sku_data["OO"].fillna(0))
    buffer_qty = df_sku_data["bufferConsumption"] - (df_sku_data["OH"].fillna(0) + df_sku_data["OO"].fillna(0))

    df_sku_data["baseDemand"] = np.maximum(base_qty, 0)
    df_sku_data["bufferDemand"] = np.maximum(buffer_qty, 0)
    df_sku_data["baseDemand"] = np.minimum(df_sku_data["baseDemand"], df_sku_data["Planned_Demand"].fillna(0))
    df_sku_data["excess_demand"] = np.maximum(df_sku_data["Planned_Demand"].fillna(0) - df_sku_data["bufferDemand"], 0)

    df_sku_data["baseDemand"] = np.where(
        df_sku_data["baseConsumption"] == 0,
        df_sku_data["Planned_Demand"],             # if baseConsumption == 0 → use planned_demand
        df_sku_data["baseDemand"]                  # otherwise keep existing baseDemand
    )

    #snapping baseDemand to MOQ, since its not really a choice
    df_sku_data["baseDemand"] = np.maximum(df_sku_data["baseDemand"], df_sku_data["CHW_MOQ_LEVEL"])

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

    #df_sku_data = process_demand_data()
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


    
