# disk_state_store_simple.py
from __future__ import annotations
import json, os, tempfile, gzip, sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime  # used by save_vendor_state_blob
from pathlib import Path
from typing import Any, Optional, List

# Database path (same as sql_lite_store.py)
ROOT = Path(__file__).resolve().parents[1]
LOCAL_DB_PATH = Path(ROOT, "data", "inventory_data.db")

VENDOR_STATE_TABLE = "vendor_states"

def save_vendor_state_blob(
    base_dir: str,
    vendor_state: Any,
    *,
    gzip_compress: bool = False,
    versioned: bool = True,   # also keep a timestamped snapshot
) -> dict:
    """
    Save the entire vendor_state as a single JSON (or .json.gz) file.
    Returns paths written.
    """
    vc = _get(vendor_state, "vendor_Code", "unknown")
    vendor_dir = os.path.join(base_dir, "data/vendor_plans", str(vc))
    os.makedirs(vendor_dir, exist_ok=True)

    # file names
    ext = ".json.gz" if gzip_compress else ".json"
    latest_path = os.path.join(vendor_dir, f"vendor_state{ext}")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    #snap_path = os.path.join(vendor_dir, f"vendor_state_{ts}{ext}") if versioned else None
    snap_path = os.path.join(vendor_dir, f"vendor_state_{ext}") if versioned else None

    payload = _to_jsonable(vendor_state)

    _atomic_dump(payload, latest_path, gzip_compress=gzip_compress)

    return {"latest": latest_path, "snapshot": snap_path}

def load_vendor_state_blob(path: str) -> Any:
    """
    Load a previously saved vendor_state blob.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def latest_vendor_state_path(base_dir: str, vendor_code: str, *, gzip_compress: bool = False) -> Optional[str]:
    ext = ".json.gz" if gzip_compress else ".json"
    p = os.path.join(base_dir, "state", str(vendor_code), f"vendor_state{ext}")
    return p if os.path.exists(p) else None

# ---------------- helpers ----------------

def _atomic_dump(obj: Any, path: str, *, gzip_compress: bool) -> None:
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirpath)
    try:
        if gzip_compress:
            with gzip.open(fd, "wt", encoding="utf-8") as f:  # open file descriptor via gzip
                json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def _to_jsonable(x: Any) -> Any:
    if x is None: return None
    if hasattr(x, "model_dump"): return x.model_dump()
    if hasattr(x, "dict"): return x.dict()
    if is_dataclass(x): return asdict(x)
    if isinstance(x, (list, dict, tuple, str, int, float, bool)): return x
    if hasattr(x, "__dict__"):  # recurse fields
        return {k: _to_jsonable(v) for k, v in x.__dict__.items()}
    return str(x)

def _get(o: Any, k: str, default=None):
    return getattr(o, k, o.get(k, default) if isinstance(o, dict) else default)


# ---------------- Database Functions ----------------

def _db_connect():
    """Create a connection to the inventory database."""
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_vendor_state_table(conn: sqlite3.Connection) -> None:
    """Create the vendor_states table if it doesn't exist."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {VENDOR_STATE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_code TEXT NOT NULL,
            vendor_state_json TEXT NOT NULL
        )
    """)
    # Index for faster lookups by vendor_code
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_vendor_code 
        ON {VENDOR_STATE_TABLE} (vendor_code)
    """)
    conn.commit()


def save_vendor_state_to_db(
    vendor_state: Any,
    *,
    wipe_existing: bool = False,
    vendor_code: Optional[str] = None,
) -> dict:
    """
    Save a vendor_state object to the inventory_data.db database.
    
    Args:
        vendor_state: The vendor state object to save. Can be a dataclass, 
                      Pydantic model, dict, or any object with __dict__.
        wipe_existing: If True, clears ALL existing records in the table before inserting.
                       If False (default), appends to the table.
        vendor_code: Optional override for vendor_code. If not provided, 
                     extracts from vendor_state.vendor_Code.
    
    Returns:
        dict with keys: 'success', 'row_id', 'vendor_code'
    """
    vc = vendor_code or _get(vendor_state, "vendor_Code", "unknown")
    payload = _to_jsonable(vendor_state)
    json_str = json.dumps(payload, ensure_ascii=False)
    
    try:
        with _db_connect() as conn:
            _ensure_vendor_state_table(conn)
            
            if wipe_existing:
                conn.execute(f"DELETE FROM {VENDOR_STATE_TABLE}")
            
            cursor = conn.execute(
                f"""
                INSERT INTO {VENDOR_STATE_TABLE} (vendor_code, vendor_state_json)
                VALUES (?, ?)
                """,
                (str(vc), json_str)
            )
            row_id = cursor.lastrowid
            conn.commit()
            
        return {
            "success": True,
            "row_id": row_id,
            "vendor_code": str(vc),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "vendor_code": str(vc),
        }


def save_vendor_states_to_db(
    vendor_states: List[Any],
    *,
    wipe_existing: bool = False,
) -> dict:
    """
    Save multiple vendor_state objects to the database in a single transaction.
    
    Args:
        vendor_states: List of vendor state objects to save.
        wipe_existing: If True, clears ALL existing records before inserting.
                       If False (default), appends to the table.
    
    Returns:
        dict with keys: 'success', 'inserted_count', 'vendor_codes'
    """
    if not vendor_states:
        return {"success": True, "inserted_count": 0, "vendor_codes": []}
    
    records = []
    vendor_codes = []
    
    for vs in vendor_states:
        vc = _get(vs, "vendor_Code", "unknown")
        payload = _to_jsonable(vs)
        json_str = json.dumps(payload, ensure_ascii=False)
        records.append((str(vc), json_str))
        vendor_codes.append(str(vc))
    
    try:
        with _db_connect() as conn:
            _ensure_vendor_state_table(conn)
            
            if wipe_existing:
                conn.execute(f"DELETE FROM {VENDOR_STATE_TABLE}")
            
            conn.executemany(
                f"""
                INSERT INTO {VENDOR_STATE_TABLE} (vendor_code, vendor_state_json)
                VALUES (?, ?)
                """,
                records
            )
            conn.commit()
            
        return {
            "success": True,
            "inserted_count": len(records),
            "vendor_codes": vendor_codes,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "inserted_count": 0,
        }


def load_vendor_state_from_db(
    vendor_code: Optional[str] = None,
    *,
    latest_only: bool = True,
    limit: Optional[int] = None,
) -> List[dict]:
    """
    Load vendor_state object(s) from the database.
    
    Args:
        vendor_code: If provided, filters by this vendor_code. 
                     If None, returns states for all vendors.
        latest_only: If True (default), returns only the most recent record per vendor.
                     If False, returns all records (subject to limit).
        limit: Maximum number of records to return. None means no limit.
    
    Returns:
        List of dicts, each containing: 'id', 'vendor_code', 'vendor_state'
        The 'vendor_state' field contains the deserialized JSON object.
    """
    results = []
    
    try:
        with _db_connect() as conn:
            _ensure_vendor_state_table(conn)
            
            if latest_only:
                # Use a subquery to get the max id (most recent) per vendor_code
                if vendor_code:
                    query = f"""
                        SELECT id, vendor_code, vendor_state_json
                        FROM {VENDOR_STATE_TABLE}
                        WHERE vendor_code = ? 
                        ORDER BY id DESC
                        LIMIT 1
                    """
                    cursor = conn.execute(query, (str(vendor_code),))
                else:
                    query = f"""
                        SELECT id, vendor_code, vendor_state_json
                        FROM {VENDOR_STATE_TABLE} v1
                        WHERE id = (
                            SELECT MAX(id) FROM {VENDOR_STATE_TABLE} v2 
                            WHERE v2.vendor_code = v1.vendor_code
                        )
                        ORDER BY vendor_code
                    """
                    if limit:
                        query += f" LIMIT {int(limit)}"
                    cursor = conn.execute(query)
            else:
                # Return all records
                if vendor_code:
                    query = f"""
                        SELECT id, vendor_code, vendor_state_json
                        FROM {VENDOR_STATE_TABLE}
                        WHERE vendor_code = ?
                        ORDER BY id DESC
                    """
                    if limit:
                        query += f" LIMIT {int(limit)}"
                    cursor = conn.execute(query, (str(vendor_code),))
                else:
                    query = f"""
                        SELECT id, vendor_code, vendor_state_json
                        FROM {VENDOR_STATE_TABLE}
                        ORDER BY id DESC
                    """
                    if limit:
                        query += f" LIMIT {int(limit)}"
                    cursor = conn.execute(query)
            
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "vendor_code": row[1],
                    "vendor_state": json.loads(row[2]),
                })
                
    except Exception as e:
        print(f"[load_vendor_state_from_db] Error: {e}")
        return []
    
    return results


def delete_vendor_states_from_db(
    vendor_code: Optional[str] = None,
    *,
    delete_all: bool = False,
) -> dict:
    """
    Delete vendor_state records from the database.
    
    Args:
        vendor_code: If provided, deletes only records for this vendor.
        delete_all: If True, deletes ALL records (ignores vendor_code).
                    Must be explicitly set to True to delete all.
    
    Returns:
        dict with keys: 'success', 'deleted_count'
    """
    try:
        with _db_connect() as conn:
            _ensure_vendor_state_table(conn)
            
            if delete_all:
                cursor = conn.execute(f"DELETE FROM {VENDOR_STATE_TABLE}")
            elif vendor_code:
                cursor = conn.execute(
                    f"DELETE FROM {VENDOR_STATE_TABLE} WHERE vendor_code = ?",
                    (str(vendor_code),)
                )
            else:
                return {
                    "success": False,
                    "error": "Must specify vendor_code or set delete_all=True",
                    "deleted_count": 0,
                }
            
            deleted_count = cursor.rowcount
            conn.commit()
            
        return {"success": True, "deleted_count": deleted_count}
    except Exception as e:
        return {"success": False, "error": str(e), "deleted_count": 0}


def list_vendor_codes_in_db() -> List[str]:
    """
    List all distinct vendor codes stored in the database.
    
    Returns:
        List of vendor_code strings.
    """
    try:
        with _db_connect() as conn:
            _ensure_vendor_state_table(conn)
            cursor = conn.execute(
                f"SELECT DISTINCT vendor_code FROM {VENDOR_STATE_TABLE} ORDER BY vendor_code"
            )
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"[list_vendor_codes_in_db] Error: {e}")
        return []


def drop_vendor_state_table() -> dict:
    """
    Drop the entire vendor_states table from the database.
    
    WARNING: This permanently deletes the table and all its data.
    
    Returns:
        dict with keys: 'success', 'message' or 'error'
    """
    try:
        with _db_connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {VENDOR_STATE_TABLE}")
            conn.commit()
        return {
            "success": True,
            "message": f"Table '{VENDOR_STATE_TABLE}' dropped successfully.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }