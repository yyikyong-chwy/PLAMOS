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




#putting them together
if __name__ == "__main__":

    config = get_snowflake_config()
    conn = setconnection(config)

    df_sku_data = process_demand_data()
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")
    df_kepplerSplits = snowflake_pull.run_query_to_df(conn, snowflake_pull.SQL_KEPLER_SPLITS)

    #generating state objects for langgraph workflow
    sku_data_state_list = state_loader.df_to_chewy_sku_states(df_sku_data, df_kepplerSplits)
    vendor_state_list = state_loader.df_to_vendor_states(df_sku_data, df_CBM_Max, sku_data_state_list)

    print(vendor_state_list)
