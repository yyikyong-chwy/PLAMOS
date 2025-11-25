# disk_state_store_simple.py
from __future__ import annotations
import json, os, tempfile, gzip
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Optional

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
    vendor_dir = os.path.join(base_dir, "data", str(vc))
    os.makedirs(vendor_dir, exist_ok=True)

    # file names
    ext = ".json.gz" if gzip_compress else ".json"
    latest_path = os.path.join(vendor_dir, f"vendor_state{ext}")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    #snap_path = os.path.join(vendor_dir, f"vendor_state_{ts}{ext}") if versioned else None
    snap_path = os.path.join(vendor_dir, f"vendor_state_{ext}") if versioned else None

    payload = _to_jsonable(vendor_state)

    _atomic_dump(payload, latest_path, gzip_compress=gzip_compress)
    if versioned and snap_path:
        _atomic_dump(payload, snap_path, gzip_compress=gzip_compress)

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
