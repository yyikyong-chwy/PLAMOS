# data/sqlite_store.py
import sqlite3
import pandas as pd
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path (same as your current pattern)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One DB file, many tables
LOCAL_DB_PATH = Path(ROOT, "data", "inventory_data.db")
LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# --- Utilities ---
def _connect():
    # WAL improves concurrency; harmless for simple apps
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def list_tables():
    """Return a list of user tables in the database."""
    with _connect() as conn:
        q = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        return [r[0] for r in conn.execute(q).fetchall()]

def drop_table(table_name: str):
    """Drop a table if it exists."""
    _validate_ident(table_name)
    with _connect() as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')


def _validate_ident(name: str):
    # Very small safety: allow letters, numbers, underscore; quote when used.
    import re
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid table name: {name!r}")


# --- Core API ---
def init_database():
    """No-op placeholder; DB file is created on first write."""
    LOCAL_DB_PATH.touch(exist_ok=True)

def save_table(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",   # "fail" | "replace" | "append"
    add_timestamp: bool = False,
    index: bool = False,
):
    """
    Save a DataFrame to a specific table.

    - if_exists='replace' (default) recreates the table each time (like your current code).
    - use 'append' to accumulate rows over time.
    """
    _validate_ident(table_name)
    if df is None or df.empty:
        return False, 0

    data_to_save = df.copy()
    # if add_timestamp and "upload_timestamp" not in data_to_save.columns:
    #     data_to_save["upload_timestamp"] = pd.Timestamp.now()

    try:
        with _connect() as conn:
            cur = conn.cursor()

            # If we're replacing, drop BOTH table and view if they exist
            if if_exists == "replace":
                # Remove any object with this name, regardless of type
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                cur.execute(f'DROP VIEW IF EXISTS "{table_name}"')
                conn.commit()
                # After explicit drop, create as new
                data_to_save.to_sql(table_name, conn, if_exists="fail", index=index)
            else:
                # Append or fail — rely on pandas
                data_to_save.to_sql(table_name, conn, if_exists=if_exists, index=index)

        return True, len(data_to_save)
    except Exception as e:
        print(f"[save_table] if_exists={if_exists}, table={table_name}, error={e}")
        return False, 0


def load_table(
    table_name: str,
    columns: list[str] | None = None,
    where: str | None = None,
    order_by: str | None = None,
    limit: int | None = None,
):
    """
    Load rows from a table. `where`, `order_by` are raw SQL fragments (optional).
    NOTE: Only pass trusted strings to `where`/`order_by`. For user inputs, parameterize instead.
    """
    _validate_ident(table_name)

    sel = "*"
    if columns:
        # minimal quoting of identifiers
        for c in columns:
            if not isinstance(c, str):
                raise ValueError("Column names must be strings")
        sel = ", ".join([f'"{c}"' for c in columns])

    q = [f'SELECT {sel} FROM "{table_name}"']
    if where:
        q.append(f"WHERE {where}")
    if order_by:
        q.append(f"ORDER BY {order_by}")
    if limit and isinstance(limit, int):
        q.append(f"LIMIT {limit}")
    sql = " ".join(q)

    try:
        with _connect() as conn:
            df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        if "no such table" not in str(e).lower():
            print(f"[load_table] Error: {e}")
        return pd.DataFrame()


# --- Backward-compatible wrappers (keep your old names) ---
def save_to_database(df: pd.DataFrame):
    """Old API: save to fixed table 'inventory_data' (replaces table)."""
    return save_table(df, table_name="inventory_data", if_exists="replace")

def load_from_database():
    """Old API: load all from fixed table 'inventory_data'."""
    return load_table("inventory_data")
