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
WITH base AS (
  SELECT
    a.item_id AS CHW_SKU_NUMBER,
    (CASE WHEN b.cartonization_flag THEN 'true' ELSE 'false' END) AS cartonization_flag,
    SUM(NULLIF(a.outflow_quantity, 0)) AS Total_Stat_FCAST,
    SUM(CASE WHEN a.location_code IN ('CFC1','DAY1','DFW1','MCI1','PHX1','RNO1','HOU1')
             THEN NULLIF(a.outflow_quantity,0) ELSE 0 END) AS TLA1_Fcast,
    SUM(CASE WHEN a.location_code IN ('AVP1','AVP2','CLT1','MCO1','BNA1')
             THEN NULLIF(a.outflow_quantity,0) ELSE 0 END) AS TNY1_Fcast,
    SUM(CASE WHEN a.location_code IN ('MDT1')
             THEN NULLIF(a.outflow_quantity,0) ELSE 0 END) AS MDT1_Fcast
  FROM edldb.sc_replenishment_sandbox.replenishment_inventory_recommended_outflow a
  JOIN edldb.chewybi.products b
    ON a.item_id = b.product_part_number
  WHERE outflow_snapshot_date = (
          SELECT MAX(outflow_snapshot_date)
          FROM edldb.sc_replenishment_sandbox.replenishment_inventory_recommended_outflow
        )
    AND b.product_discontinued_flag = 'false'
    AND b.private_label_flag = 'true'
    AND forecast_date = CURRENT_DATE
    AND a.location_code IN (
      'AVP1','AVP2','BNA1','CFC1','CLT1','DAY1','DFW1','MCI1','MCO1','MDT1','PHX1','RNO1','HOU1'
    )
  GROUP BY 1, 2
)
SELECT
  CHW_SKU_NUMBER AS ITEM_ID,
  cartonization_flag,
  Total_Stat_FCAST,
  TLA1_Fcast,
  TNY1_Fcast,
  MDT1_Fcast,
  /* Fractions: if total is 0, return 0; otherwise divide */
  IFF(Total_Stat_FCAST = 0, 0, TLA1_Fcast / Total_Stat_FCAST) AS TLA1_FRAC,
  IFF(Total_Stat_FCAST = 0, 0, TNY1_Fcast / Total_Stat_FCAST) AS TNY1_FRAC,
  IFF(Total_Stat_FCAST = 0, 0, MDT1_Fcast / Total_Stat_FCAST) AS MDT1_FRAC
FROM base;
"""


SQL_SKU_MARGIN = """
SELECT 
    pp.product_part_number,

    /* SKU-level units + sales */
    SUM(COALESCE(olcm.order_line_quantity, 0))              AS units,
    SUM(COALESCE(olcm.order_line_total_price, 0))           AS total_price,
    SUM(COALESCE(olcm.order_line_net_sales, 0))             AS net_sales,  -- topline + adjustment + ship charge

    /* === SKU-level COGS === */
    SUM(
        COALESCE(olcm.order_line_raw_product_margin, 0)
        - COALESCE(olcm.order_line_net_sales, 0)
        - COALESCE(olcm.order_line_total_standard_inbound_cost, 0)
    ) AS cogs,  -- CoGS + royalties (per your original logic)

    /* === SKU-level product margin === */
    SUM(
        COALESCE(olcm.order_line_raw_product_margin, 0)
    ) AS product_margin,

    /* product margin per unit */
    SUM(COALESCE(olcm.order_line_raw_product_margin, 0))
        / NULLIF(SUM(COALESCE(olcm.order_line_quantity, 0)), 0) AS product_margin_per_unit,

    /* inbound freight */
    SUM(COALESCE(olcm.order_line_total_standard_inbound_cost, 0)) AS inbound_freight,

    /* subtotal = COGS + inbound freight, same as original logic */
    (
        SUM(
            COALESCE(olcm.order_line_raw_product_margin, 0)
            - COALESCE(olcm.order_line_net_sales, 0)
            - COALESCE(olcm.order_line_total_standard_inbound_cost, 0)
        )
        + SUM(COALESCE(olcm.order_line_total_standard_inbound_cost, 0))
    ) AS cogs_subtotal

FROM edldb.chewybi.order_line_cost_measures AS olcm
INNER JOIN edldb.chewybi.orders AS orders USING (order_key)
INNER JOIN edldb.chewybi.products pp 
    ON pp.product_key = olcm.product_key
LEFT JOIN edldb.chewybi.business_channels 
    ON business_channels.business_channel_key = olcm.business_channel_key
WHERE 
    olcm.product_company_description = 'Chewy'
    AND LOWER(orders.order_status) NOT IN ('x', 'j')
    AND olcm.order_placed_date::date BETWEEN CURRENT_DATE - 720 AND CURRENT_DATE - 1
    AND pp.product_merch_classification1 NOT IN ('Gift Cards', 'Virtual Bundle', 'Programs')
    AND pp.product_merch_classification1 IS NOT NULL
    AND pp.product_discontinued_flag = 'false'
    AND UPPER(TRIM(pp.product_merch_classification2)) <> 'PHARMACY'
    AND pp.private_label_flag = 'true'
GROUP BY 
    pp.product_part_number;
"""

SQL_Vendor_CBM = """
select PRODUCT, CHW_SKU_NUMBER, MC1_NAME, MC2_NAME, MC3_NAME, Brand, CUSTOMER_EARLIEST_TARGET_DATE, EARLIEST_TARGET_DATE, CHW_MOQ_LEVEL,
CHW_OTB, CHW_PRIMARY_SUPPLIER_NAME, CHW_PRIMARY_SUPPLIER_NUMBER, PRIMARY_SUPPLIER, CHW_MASTER_CASE_PACK, CHW_MASTER_CARTON_CBM
from edldb.mrch_portfolio_sandbox.plm_sku_snapshot
where active = 1
and CHW_PRIMARY_SUPPLIER_NAME is not null
and nullif(trim(CHW_SKU_NUMBER), '') is not null
"""

SQL_SKU_FCST = """
select c.product_part_number as SKU, c.forecast_date, c.dw_fcst
FROM edldb.sc_sandbox.DW_fcst_item_day_network_colt c
JOIN edldb.chewybi.products p ON c.PRODUCT_PART_NUMBER = p.PRODUCT_PART_NUMBER 
join chewybi.product_attributes pa on pa.partnumber = p.product_part_number        
WHERE 1=1 
and COALESCE(pa."attribute.onetimebuy", 'FALSE') != 'TRUE'
AND SNAPSHOT_DATE = (select max(SNAPSHOT_DATE) from edldb.sc_sandbox.DW_fcst_item_day_network_colt)
and p.private_label_flag = '1'
and p.product_discontinued_flag = 'false'
AND c.FORECAST_DATE BETWEEN current_date and current_date+180
AND p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')
and COALESCE(pa."attribute.onetimebuy", 'FALSE') != 'TRUE';
"""


SQL_SKU_Supply_Snapshot = """

with T90 as (        
        select 
        p.product_part_number as SKU
        ,round(sum(olm.order_line_quantity)/90,2) as T90_Daily_Avg
        
        from chewybi.order_line_measures olm
        join chewybi.products p on p.product_key = olm.product_key
        join chewybi.product_attributes pa on pa.partnumber = p.product_part_number

        where 1=1
        and COALESCE(pa."attribute.onetimebuy", 'FALSE') != 'TRUE'
        and p.private_label_flag = '1'
        --and p.product_merch_classification1 in ('Hard Goods','Dog Consumables','Cat Consumables')
        and p.product_discontinued_flag = 'false'
        and olm.order_line_released_dttm is not null
        and (date(olm.order_placed_date) between current_date-90 and current_date-1)
        AND p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')
        AND order_line_each_price <> 0 

        group by 1
        order by 1 asc
),       
 
F90 as (
        SELECT  p.PRODUCT_PART_NUMBER as SKU, 
        round(SUM(DW_FCST)/90,2) as F90_Daily_Avg
        
        FROM edldb.sc_sandbox.DW_fcst_item_day_network_colt c

        JOIN edldb.chewybi.products p ON c.PRODUCT_PART_NUMBER = p.PRODUCT_PART_NUMBER 
        join chewybi.product_attributes pa on pa.partnumber = p.product_part_number
        
        WHERE 1=1 
        and COALESCE(pa."attribute.onetimebuy", 'FALSE') != 'TRUE'
        AND SNAPSHOT_DATE = (select max(SNAPSHOT_DATE) from edldb.sc_sandbox.DW_fcst_item_day_network_colt)
        and p.private_label_flag = '1'
        --and p.product_merch_classification1 in ('Hard Goods','Dog Consumables','Cat Consumables')
        and p.product_discontinued_flag = 'false'
        AND c.FORECAST_DATE BETWEEN current_date and current_date+90
        AND p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')

        GROUP BY 1
),

F180 as (
        SELECT  p.PRODUCT_PART_NUMBER as SKU, 
        round(SUM(DW_FCST)/180,2) as F180_Daily_Avg
        
        FROM edldb.sc_sandbox.DW_fcst_item_day_network_colt c

        JOIN edldb.chewybi.products p ON c.PRODUCT_PART_NUMBER = p.PRODUCT_PART_NUMBER 
        join chewybi.product_attributes pa on pa.partnumber = p.product_part_number
        
        WHERE 1=1 
        and COALESCE(pa."attribute.onetimebuy", 'FALSE') != 'TRUE'
        AND SNAPSHOT_DATE = (select max(SNAPSHOT_DATE) from edldb.sc_sandbox.DW_fcst_item_day_network_colt)
        and p.private_label_flag = '1'
        and p.product_discontinued_flag = 'false'
        AND c.FORECAST_DATE BETWEEN current_date and current_date+180
        AND p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')

        GROUP BY 1
),

OH as (
        select
        p.product_part_number,
        zeroifnull(sum(inventory_snapshot_sellable_quantity)) as current_on_hand

        from edldb.chewybi.products p
        left join edldb.chewybi.inventory_snapshot i 
                on i.product_key = p.product_key
        left join edldb.chewybi.locations l 
                on l.location_key = i.location_key
        join chewybi.product_attributes pa 
                on pa.partnumber = p.product_part_number
                and pa."attribute.onetimebuy" = 'false'
        
        where 1=1
        and inventory_snapshot_snapshot_dt = (select max(inventory_snapshot_snapshot_dt) from edldb.chewybi.inventory_snapshot)
        and l.fulfillment_active in ('true')
        and p.private_label_flag = 'true'
        --and p.product_merch_classification1 in ('Hard Goods','Dog Consumables','Cat Consumables')
        and p.product_discontinued_flag = 'false'
        and l.location_active_warehouse = 1
        and l.location_warehouse_type = 0
        AND p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')
        
        group by 1
),

LT as (
        select 
        bpa.vendor_number as vendor_number,
        vendor_name,
        category_id,
        p.product_part_number,
        p.product_merch_classification1,
        case
                when category_id = 'Shipping Container'
                and product_merch_classification1 = 'Hard Goods'
                then round(avg(avg_leadtime),0)+14 
                else round(avg(avg_leadtime),0) 
        end as avg_lt

        from chewybi.vendor_agreement_line bpa
        join chewybi.products p
                on p.product_part_number = bpa.product_part_number
        join sc_replenishment_sandbox.SOCRATES_LEAD_TIME_PREDICTION_INPUT lt
                on bpa.vendor_number = lt.vendor_id
        join chewybi.product_attributes pa 
                on pa.partnumber = p.product_part_number

        where 1=1
        and job_start_time = (
                SELECT MAX(job_start_time)
                FROM edldb.sc_replenishment_sandbox.SOCRATES_LEAD_TIME_PREDICTION_INPUT
                )
        and active_agreement =1
        and COALESCE(pa."attribute.onetimebuy", 'FALSE') != 'TRUE'
        and p.private_label_flag = '1'
        --and p.product_merch_classification1 in ('Hard Goods','Dog Consumables','Cat Consumables')
        and p.product_discontinued_flag = 'false'
        and p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')
        
        group by 1,2,3,4,5
),

OO as (
        select  
        pdpm.product_part_number as SKU, 
        sum(pdpm.outstanding_quantity) as OO,
        min(pdpm.document_requested_delivery_dttm) as next_delivery

        from chewybi.procurement_document_product_measures pdpm
        join   chewybi.locations l
                on pdpm.location_key = l.location_key 
        join chewybi.vendors v
                on pdpm.vendor_key = v.vendor_key
        join chewybi.products p
                on pdpm.product_part_number = p.product_part_number
        
        where 1=1
        and pdpm.document_order_dttm > current_date-120
        and pdpm.DOCUMENT_READY_TO_RECONCILE_FLAG = 'false'
        and document_status not in ('CANCELLED','REJECTED','CLOSED','CLOSED FOR RECEIVING','INCOMPLETE','FINALLY CLOSED')
        and DOCUMENT_WMS_CLOSED_FLAG = 'false'
        and DELETED_BY_USERS = 'false'
        and p.private_label_flag = 'true'
        --and p.product_merch_classification1 in ('Hard Goods','Dog Consumables','Cat Consumables')
        and pdpm.document_number not like ('TR%')
        and p.product_discontinued_flag = 'false'

        group by 1
)

        select
        p.product_merch_classification1 as MC1,
        p.product_merch_classification2 as MC2,
        p.product_manufacturer_name as Brand,
        p.parent_product_part_number as pSku,
        p.product_part_number as sku, 
        p.product_name,
        product_published_flag as Pubbed,
        current_on_hand as OH,
        T90_Daily_Avg,
        F90_Daily_Avg,
        avg_lt,
        zeroifnull(OO) as OO,
        Next_delivery,
        
        case when T90_Daily_Avg = 0
        then null
        else
        round(sum(inventory_snapshot_sellable_quantity)/ifnull(T90_Daily_Avg,.00001),2) end as T90_DOS_OH,
        
        case when F90_Daily_Avg = 0
        then null
        else
        round(sum(inventory_snapshot_sellable_quantity)/ifnull(F90_Daily_Avg,.00001),2) end as F90_DOS_OH,
        
        case when F90_Daily_Avg = 0
        then null
        else
        round(zeroifnull(OO)/F90_Daily_Avg,2) end as F90_DOS_OO,
        
        case when F180_Daily_Avg = 0
        then null
        else
        round(sum(inventory_snapshot_sellable_quantity)/ifnull(F180_Daily_Avg,.00001),2) end as F180_DOS_OH,
        
        case when F180_Daily_Avg = 0
        then null
        else
        round(zeroifnull(OO)/F180_Daily_Avg,2) end as F180_DOS_OO,
        
        case when T90_DOS_OH < avg_lt
                then true
                else false
                end as T90_below,
        case when F90_DOS_OH < avg_lt
                then true
                else false
                end as F90_below,
        case when F90_below =1
                and (next_delivery is null
                or next_delivery < current_date)
                then true
                else false
                end as Alert
        
        from edldb.chewybi.products p
        join OH
                on p.product_part_number = OH.product_part_number
        left join edldb.chewybi.inventory_snapshot i 
                on i.product_key = p.product_key
        left join edldb.chewybi.locations l 
                on l.location_key = i.location_key
        join chewybi.product_attributes pa 
                on pa.partnumber = p.product_part_number
        join T90
                on p.product_part_number = T90.SKU
        join F90
                on p.product_part_number = F90.SKU
        join F180
                on p.product_part_number = F180.SKU
        join LT
                on LT.product_part_number = p.product_part_number
        left join OO
                on p.product_part_number = OO.SKU                                
                        
        where 1=1
        and pa."attribute.onetimebuy" = 'false'
        and inventory_snapshot_snapshot_dt = (select max(inventory_snapshot_snapshot_dt) from edldb.chewybi.inventory_snapshot)
        and l.fulfillment_active in ('true')
        and p.private_label_flag = 'true'
        --and p.product_merch_classification1 in ('Hard Goods','Dog Consumables','Cat Consumables')
        and p.product_discontinued_flag = 'false'
        and l.location_active_warehouse = 1
        and l.location_warehouse_type = 0
        AND p.PRODUCT_COMPANY_DESCRIPTION in ('Chewy')
          GROUP BY
        p.product_merch_classification1,
        p.product_merch_classification2,
        p.product_manufacturer_name,
        p.parent_product_part_number,
        p.product_part_number,
        p.product_name,
        product_published_flag,
        OH.current_on_hand,
        T90.T90_Daily_Avg,
        F90.F90_Daily_Avg,
        F180.F180_Daily_Avg,
        LT.avg_lt,
        zeroifnull(OO.OO),
        OO.next_delivery
        order by 1
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

        df_skuSupplySnapshot = run_query_to_df(conn, SQL_SKU_Supply_Snapshot)

        print(df_skuSupplySnapshot.head())



    finally:
        # Close the shared connection when you're done
        conn.close()
