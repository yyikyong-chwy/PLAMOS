import pandas as pd
import sqlite3
import sys
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from data.snowflake_pull import get_snowflake_config, setconnection, run_query_to_df
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
    df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")

    config = get_snowflake_config()
    conn = setconnection(config)

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

    df_demand = df_demand.merge(
        df_vendor_cbm,
        how='left',
        on=['CHW_SKU_NUMBER'])

    #combine the 2 tables,
    df_demand['CHW_SKU_NUMBER'] = df_demand['CHW_SKU_NUMBER'].astype(str).str.strip()
    df_skuSupplySnapshot['SKU']= df_skuSupplySnapshot['SKU'].astype(str).str.strip()
    #detected duplicated SKU in df_skuSupplySnapshot, keep the first max AVG_LT
    df_skuSupplySnapshot = keep_first_max_avglt(df_skuSupplySnapshot)


    df_sku_data = df_demand.merge(
        df_skuSupplySnapshot,
        how='left',
        left_on='CHW_SKU_NUMBER',
        right_on='SKU'
    )

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

    return df_sku_data


def split_base_demand_by_dest(df_sku_data: pd.DataFrame,
                              df_kepplerSplits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge df_sku_data with df_kepplerSplits on CHW_SKU_NUMBER == ITEM_ID (or item_id),
    then split baseDemand into TLA1/TNY1/MDT1 using *_FRAC.
    If no match in df_kepplerSplits, assign all demand to TNY1 and 0 to the others.

    Output: long dataframe with a DEST column and baseDemand_dest (per-destination).
    """
    columns_to_keep = ['ITEM_ID', 'TLA1_FRAC', 'TNY1_FRAC', 'MDT1_FRAC']  # specify which columns you want
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

    # Also fill any remaining NaNs (e.g., partial data quality) with 0
    merged[["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]] = \
        merged[["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]].fillna(0.0)

    # --- long form: DEST + frac ---
    long = merged.melt(
        id_vars=merged.columns.difference(["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"]),
        value_vars=["TLA1_FRAC", "TNY1_FRAC", "MDT1_FRAC"],
        var_name="DEST",
        value_name="frac"
    )
    long["DEST"] = long["DEST"].str.replace("_FRAC", "", regex=False)

    # --- compute split demand ---
    long["Planned_Demand"] = long["Planned_Demand"].fillna(0.0).astype(float)
    long["Planned_Demand_dest"] = long["Planned_Demand"] * long["frac"]

    #snapping baseDemand_dest to MCP
    mcp = pd.to_numeric(long["CHW_MASTER_CASE_PACK"], errors="coerce")
    demand_units = pd.to_numeric(long["Planned_Demand_dest"], errors="coerce").fillna(0)

    # Avoid division by zero
    safe_mcp = mcp.replace(0, np.nan)
    cases = (demand_units / safe_mcp).round()          # nearest integer #cases
    cases = cases.fillna(0).clip(lower=0).astype("Int64")
    long["Planned_Demand_cases_need"] = cases



    return long


#putting them together
if __name__ == "__main__":

    config = get_snowflake_config()
    conn = setconnection(config)

    #df_sku_data = process_demand_data()
    df_sku_data = sql_lite_store.load_table("df_sku_data")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")
    df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")
    #df_kepplerSplits = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)

    demand_by_Dest = split_base_demand_by_dest(df_sku_data, df_kepplerSplits)

    #generating state objects for langgraph workflow
    sku_data_state_list = state_loader.df_to_chewy_sku_states(df_sku_data)
    container_plan_rows = state_loader.load_container_plan_rows(demand_by_Dest)

    vendor_state_list = state_loader.df_to_vendor_states(df_sku_data, df_CBM_Max, sku_data_state_list, container_plan_rows)


    
