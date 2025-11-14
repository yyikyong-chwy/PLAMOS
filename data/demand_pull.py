# pip install snowflake-connector-python pandas
import os
import pandas as pd
import snowflake.connector as sf
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# # # ---- Snowflake connection details (edit or use env vars) ----
# # # Prefer your org-account or locator.region form
# SNOWFLAKE_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT",  "chewy-chewy")   # or "chewy.us-east-1"
# SNOWFLAKE_USER      = os.getenv("SNOWFLAKE_USER",     "yyikyong@chewy.com")
# SNOWFLAKE_ROLE      = os.getenv("SNOWFLAKE_ROLE",     "SC_USER")
# SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE","SC_WH")
# SNOWFLAKE_DATABASE  = os.getenv("SNOWFLAKE_DATABASE", "EDLDB")
# SNOWFLAKE_SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA",   "SC_REPLENISHMENT_SANDBOX")

# # Auth: use externalbrowser if you sign in with SSO
# AUTHENTICATOR = os.getenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")  # or "snowflake"

def get_snowflake_config():
    """Get Snowflake configuration from secrets or environment variables"""
    try:
        # Try to get from Streamlit secrets first
        return {
            'account': os.getenv("SNOWFLAKE_ACCOUNT"),
            'user': os.getenv("SNOWFLAKE_USER"),
            'role': os.getenv("SNOWFLAKE_ROLE"),
            'warehouse': os.getenv("SNOWFLAKE_WAREHOUSE"),
            'database': os.getenv("SNOWFLAKE_DATABASE"),
            'schema': os.getenv("SNOWFLAKE_SCHEMA"),
            'authenticator': os.getenv("AUTHENTICATOR"),
        }
    except:
        # Fallback to empty config if secrets not available
        return {
            'account': "",
            'user': "",
            'role': "",
            'warehouse': "",
            'database': "",
            'schema': "",
            'authenticator': "externalbrowser",
        }


SQL_KEPLER_SPLITS = """
SELECT  a.item_id,  
(CASE WHEN b.cartonization_flag THEN 'true' ELSE 'false' END) AS  cartonization_flag,
SUM(nullif(a.outflow_quantity,0)) AS Total_Stat_FCAST,SUM(CASE WHEN a.location_code IN ('CFC1', 'DAY1', 'DFW1', 'MCI1', 'PHX1', 'RNO1','HOU1') THEN nullif(a.outflow_quantity,0) ELSE 0 END)  AS TLA1_Fcast,
SUM(CASE WHEN a.location_code IN ('AVP1', 'AVP2', 'CLT1', 'MCO1', 'BNA1') THEN nullif(a.OUTFLOW_QUANTITY,0) ELSE 0 END)  AS TNY1_Fcast,
SUM(CASE WHEN a.location_code IN ('MDT1') THEN nullif(a.OUTFLOW_QUANTITY,0) ELSE 0 END)  AS MDT1_Fcast
FROM  edldb.sc_replenishment_sandbox.replenishment_inventory_recommended_outflow a  
JOIN edldb.chewybi.products b ON a.item_id = b.product_part_number
where outflow_snapshot_date = (SELECT MAX(outflow_snapshot_date) FROM edldb.sc_replenishment_sandbox.replenishment_inventory_recommended_outflow)
and b.product_discontinued_flag = 'false' 
 AND b.private_label_flag = 'true'  
and forecast_date = current_date   
AND a.LOCATION_CODE in ('AVP1','AVP2','BNA1','CFC1','CLT1','DAY1','DFW1','MCI1','MCO1','MDT1','PHX1','RNO1','HOU1')
GROUP BY  1, 2
"""


SQL_Vendor_CBM = """
select PRODUCT, CHW_SKU_NUMBER, MC1_NAME, MC2_NAME, MC3_NAME, Brand, CUSTOMER_EARLIEST_TARGET_DATE, EARLIEST_TARGET_DATE, CHW_MOQ_LEVEL,
CHW_OTB, CHW_PRIMARY_SUPPLIER_NAME, CHW_PRIMARY_SUPPLIER_NUMBER, CHW_MASTER_CASE_PACK, CHW_MASTER_CARTON_CBM
from edldb.mrch_portfolio_sandbox.plm_sku_snapshot
where active = 1
and CHW_PRIMARY_SUPPLIER_NAME is not null
and nullif(trim(CHW_SKU_NUMBER), '') is not null
"""
def setconnection(config: dict) -> sf.SnowflakeConnection:
    conn = sf.connect(
        account=config['account'],
        user=config['user'],
        role=config['role'],
        warehouse=config['warehouse'],
        database=config['database'],
        schema=config['schema'],
        authenticator=config['authenticator'],
    )
    return conn


def run_query_to_df(connection: sf.SnowflakeConnection, sql: str) -> pd.DataFrame:
    # Reuse the provided connection; only open/close a cursor
    with connection.cursor() as cur:
        # Optional: tag the session/query
        cur.execute("ALTER SESSION SET QUERY_TAG = 'keppler_splits_pull'")
        cur.execute(sql)
        df = cur.fetch_pandas_all()
        return df

def mutate_keppler_splits(df_kepplerSplits: pd.DataFrame) -> pd.DataFrame:
    df_kepplerSplits.columns = [c.upper() for c in df_kepplerSplits.columns]
        

    df_kepplerSplits["ITEM_ID"] = df_kepplerSplits["ITEM_ID"].astype(str)

    for c in ["TOTAL_STAT_FCAST", "TLA1_FCAST", "TNY1_FCAST", "MDT1_FCAST"]:
        df_kepplerSplits[c] = pd.to_numeric(df_kepplerSplits[c], errors="coerce")

    denom = df_kepplerSplits["TOTAL_STAT_FCAST"].replace(0, np.nan)
    df_kepplerSplits["TLA1_FRAC"]  = df_kepplerSplits["TLA1_FCAST"]  / denom
    df_kepplerSplits["TNY1_FRAC"]  = df_kepplerSplits["TNY1_FCAST"]  / denom
    df_kepplerSplits["MDT1_FRAC"] = df_kepplerSplits["MDT1_FCAST"] / denom
    return df_kepplerSplits


if __name__ == "__main__":
    try:
        config = get_snowflake_config()
        conn = setconnection(config)
        df_kepplerSplits = run_query_to_df(conn, SQL_KEPLER_SPLITS)
        df_vendor_cbm = run_query_to_df(conn, SQL_Vendor_CBM)
        print(f"Rows returned: {len(df_kepplerSplits):,}")
        df_kepplerSplits = mutate_keppler_splits(df_kepplerSplits)

        print(df_kepplerSplits.head())






    finally:
        # Close the shared connection when you're done
        conn.close()
