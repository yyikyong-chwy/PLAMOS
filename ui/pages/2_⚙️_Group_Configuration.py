"""
Vendor Group Configuration Page
Configure vendor groups - add/remove vendors, create/delete groups.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Group Configuration",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------ config ------------ #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "planner_groups.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ------------ SQLite helpers ------------ #

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    # Create table if not exists
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
    return conn


def get_all_groups() -> List[str]:
    """Get list of all unique group names."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT group_name FROM vendor_group_mapping ORDER BY group_name;")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_all_vendors() -> List[str]:
    """Get list of all unique vendors across all groups."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT vendor_code FROM vendor_group_mapping ORDER BY vendor_code;")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_vendors_for_group(group_name: str) -> List[str]:
    """Get list of vendors assigned to a specific group."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT vendor_code FROM vendor_group_mapping WHERE group_name = ? ORDER BY vendor_code;",
            (group_name,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_all_group_mappings() -> Dict[str, List[str]]:
    """Get all groups and their vendors as a dictionary."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT group_name, vendor_code FROM vendor_group_mapping ORDER BY group_name, vendor_code;")
        rows = cur.fetchall()
    finally:
        conn.close()

    groups: Dict[str, List[str]] = {}
    for group_name, vendor_code in rows:
        groups.setdefault(group_name, []).append(vendor_code)
    return groups


def add_vendor_to_group(group_name: str, vendor_code: str) -> None:
    """Add a vendor to a group."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO vendor_group_mapping (group_name, vendor_code) VALUES (?, ?);",
            (group_name.strip(), vendor_code.strip()),
        )
        conn.commit()
    finally:
        conn.close()


def remove_vendor_from_group(group_name: str, vendor_code: str) -> None:
    """Remove a vendor from a group."""
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM vendor_group_mapping WHERE group_name = ? AND vendor_code = ?;",
            (group_name.strip(), vendor_code.strip()),
        )
        conn.commit()
    finally:
        conn.close()


def delete_group(group_name: str) -> None:
    """Delete an entire group and all its vendor assignments."""
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM vendor_group_mapping WHERE group_name = ?;",
            (group_name.strip(),),
        )
        conn.commit()
    finally:
        conn.close()


def create_group_with_vendors(group_name: str, vendors: List[str]) -> None:
    """Create a new group with a list of vendors."""
    conn = get_db_connection()
    try:
        records = [(group_name.strip(), v.strip()) for v in vendors if v.strip()]
        conn.executemany(
            "INSERT OR IGNORE INTO vendor_group_mapping (group_name, vendor_code) VALUES (?, ?);",
            records,
        )
        conn.commit()
    finally:
        conn.close()


# ------------ main app ------------ #

def main():
    st.title("⚙️ Vendor Group Configuration")
    st.markdown("---")

    # Load all data from database
    all_groups = get_all_groups()
    all_vendors = get_all_vendors()
    group_mappings = get_all_group_mappings()

    if not all_groups and not all_vendors:
        st.warning(
            "No data found in database. "
            "Run `python scripts/seed_vendor_planner_db.py` to initialize from CSV."
        )
        st.stop()

    # Sidebar stats
    st.sidebar.header("📊 Summary")
    st.sidebar.markdown(f"**Total Groups:** {len(all_groups)}")
    st.sidebar.markdown(f"**Total Vendors:** {len(all_vendors)}")

    # ========================================
    # SECTION 1: View & Edit Existing Groups
    # ========================================
    st.header("1. Existing Groups")

    if not all_groups:
        st.info("No groups defined yet. Create one below.")
    else:
        selected_group = st.selectbox(
            "Select a group to view/edit",
            options=all_groups,
            key="selected_group",
        )

        if selected_group:
            vendors_in_group = group_mappings.get(selected_group, [])

            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader(f"Group: `{selected_group}`")
                st.markdown(f"**Vendors in this group:** {len(vendors_in_group)}")

                if vendors_in_group:
                    st.dataframe(
                        pd.DataFrame({"vendor_code": vendors_in_group}),
                        use_container_width=True,
                        height=min(350, 40 + 30 * len(vendors_in_group)),
                    )
                else:
                    st.info("No vendors in this group.")

            with col2:
                st.subheader("Modify Group")

                # Add vendor
                st.markdown("**Add Vendor**")
                new_vendor_code = st.text_input(
                    "Vendor code to add",
                    key="add_vendor_input",
                    placeholder="e.g., B000064",
                )
                if st.button("➕ Add Vendor", key="btn_add_vendor", type="primary"):
                    if new_vendor_code.strip():
                        if new_vendor_code.strip() in vendors_in_group:
                            st.warning(f"Vendor '{new_vendor_code}' is already in this group.")
                        else:
                            add_vendor_to_group(selected_group, new_vendor_code.strip())
                            st.success(f"Added '{new_vendor_code}' to group '{selected_group}'")
                            st.rerun()
                    else:
                        st.warning("Please enter a vendor code.")

                st.markdown("---")

                # Remove vendor
                st.markdown("**Remove Vendor**")
                if vendors_in_group:
                    vendor_to_remove = st.selectbox(
                        "Select vendor to remove",
                        options=vendors_in_group,
                        key="remove_vendor_select",
                    )
                    if st.button("➖ Remove Vendor", key="btn_remove_vendor"):
                        remove_vendor_from_group(selected_group, vendor_to_remove)
                        st.success(f"Removed '{vendor_to_remove}' from group '{selected_group}'")
                        st.rerun()
                else:
                    st.info("No vendors to remove.")

                st.markdown("---")

                # Delete entire group
                st.markdown("**Delete Group**")
                if st.button(f"🗑️ Delete Entire Group", key="btn_delete_group"):
                    delete_group(selected_group)
                    st.success(f"Deleted group '{selected_group}'")
                    st.rerun()

    # ========================================
    # SECTION 2: Create New Group
    # ========================================
    st.markdown("---")
    st.header("2. Create New Group")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        new_group_name = st.text_input(
            "New group name",
            key="new_group_name_input",
            placeholder="e.g., 'NewGroup'",
        )

    with col_right:
        st.markdown("Enter vendor codes (one per line or comma-separated):")
        new_group_vendors_text = st.text_area(
            "Vendor codes",
            key="new_group_vendors_text",
            height=150,
            placeholder="B000064\nB000068\nP000009",
        )

    if st.button("Create Group", type="primary", key="btn_create_group"):
        name = new_group_name.strip()
        if not name:
            st.error("Group name cannot be empty.")
        elif name in all_groups:
            st.error(f"Group '{name}' already exists.")
        else:
            raw_vendors = new_group_vendors_text.replace(",", "\n").split("\n")
            vendors = [v.strip() for v in raw_vendors if v.strip()]

            if not vendors:
                st.error("Please enter at least one vendor code.")
            else:
                create_group_with_vendors(name, vendors)
                st.success(f"Group '{name}' created with {len(vendors)} vendors!")
                st.rerun()

    # ========================================
    # SECTION 3: All Groups Summary
    # ========================================
    st.markdown("---")
    st.header("3. All Groups Summary")

    if group_mappings:
        summary_data = []
        for group_name, vendors in sorted(group_mappings.items()):
            summary_data.append({
                "Group": group_name,
                "# Vendors": len(vendors),
                "Vendors": ", ".join(vendors[:5]) + ("..." if len(vendors) > 5 else ""),
            })

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            height=min(400, 60 + 35 * len(summary_data)),
        )
    else:
        st.info("No groups to display.")

    st.markdown("---")
    st.caption(f"✅ Changes are automatically saved to SQLite (`{DB_PATH.name}`).")


if __name__ == "__main__":
    main()

