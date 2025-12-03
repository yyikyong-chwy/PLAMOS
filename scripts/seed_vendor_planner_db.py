"""
One-time script to seed the vendor_group_mapping table from Planner_codes.csv.
Run this once to initialize the database, then vendor_group_config.py will load from DB.

CSV format:
  supplier_cd,buyer_cd
  B000064,JaSchack
  ...

Where:
  - supplier_cd = vendor_code
  - buyer_cd = group_name (e.g., JaSchack, KNgo, mwilson)

Usage:
    python scripts/seed_vendor_planner_db.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "Planner_codes.csv"
DB_PATH = PROJECT_ROOT / "data" / "planner_groups.db"


def init_schema(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vendor_group_mapping (
            group_name  TEXT NOT NULL,
            vendor_code TEXT NOT NULL,
            PRIMARY KEY (group_name, vendor_code)
        );
        """
    )
    conn.commit()


def seed_from_csv(conn: sqlite3.Connection, csv_path: Path):
    """Load vendor-group mappings from CSV into database."""
    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    df["supplier_cd"] = df["supplier_cd"].astype(str).str.strip()
    df["buyer_cd"] = df["buyer_cd"].astype(str).str.strip()

    # Filter out empty rows
    df = df[(df["supplier_cd"] != "") & (df["buyer_cd"] != "")]

    print(f"Found {len(df)} records in CSV")

    # Insert records: buyer_cd = group_name, supplier_cd = vendor_code
    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            cur.execute(
                "INSERT INTO vendor_group_mapping (group_name, vendor_code) VALUES (?, ?);",
                (row["buyer_cd"], row["supplier_cd"]),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # Already exists
            skipped += 1

    conn.commit()
    print(f"Inserted: {inserted}, Skipped (duplicates): {skipped}")
    return True


def main():
    print(f"CSV Path: {CSV_PATH}")
    print(f"DB Path: {DB_PATH}")
    print()

    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    try:
        # Initialize schema
        print("Initializing database schema...")
        init_schema(conn)

        # Check current count
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vendor_group_mapping;")
        existing_count = cur.fetchone()[0]
        print(f"Existing records in vendor_group_mapping: {existing_count}")

        if existing_count > 0:
            response = input("Database already has data. Overwrite? (y/N): ").strip().lower()
            if response == "y":
                print("Clearing existing data...")
                conn.execute("DELETE FROM vendor_group_mapping;")
                conn.commit()
            else:
                print("Keeping existing data. New records will be added if not duplicates.")

        # Seed from CSV
        print("\nSeeding from CSV...")
        seed_from_csv(conn, CSV_PATH)

        # Show summary
        cur.execute("SELECT COUNT(*) FROM vendor_group_mapping;")
        final_count = cur.fetchone()[0]
        print(f"\nFinal record count: {final_count}")

        cur.execute("SELECT DISTINCT group_name FROM vendor_group_mapping ORDER BY group_name;")
        groups = [row[0] for row in cur.fetchall()]
        print(f"Groups: {', '.join(groups)}")

        # Show vendors per group
        print("\nVendors per group:")
        for group in groups:
            cur.execute("SELECT COUNT(*) FROM vendor_group_mapping WHERE group_name = ?;", (group,))
            count = cur.fetchone()[0]
            print(f"  {group}: {count} vendors")

        print("\nDone! Database is ready.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

