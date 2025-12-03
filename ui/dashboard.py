"""
Main Dashboard for MBM Application
Navigation hub to access various tools and configurations.
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MBM Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .nav-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .nav-card:hover {
        transform: translateY(-5px);
    }
    .nav-card h3 {
        color: white;
        margin-bottom: 10px;
    }
    .nav-card p {
        color: rgba(255,255,255,0.9);
        font-size: 14px;
    }
    .main-title {
        text-align: center;
        padding: 20px 0;
    }
    .stButton > button {
        width: 100%;
        padding: 20px;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("<h1 class='main-title'>🏠 MBM Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Master Buy Management - Navigation Hub</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 25px; border-radius: 15px; color: white; height: 200px;'>
            <h3 style='color: white;'>📦 Vendor Plan Viewer</h3>
            <p style='color: rgba(255,255,255,0.9);'>
                View and analyze vendor container plans. Compare metrics across different 
                planning strategies, analyze demand fulfillment, and review container utilization.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Open Vendor Plan Viewer", key="btn_plan_viewer", use_container_width=True):
            st.switch_page("pages/1_📦_Vendor_Plan_Viewer.py")
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; color: white; height: 200px;'>
            <h3 style='color: white;'>⚙️ Vendor Group Configuration</h3>
            <p style='color: rgba(255,255,255,0.9);'>
                Configure vendor groups. Add or remove vendors from groups, create new groups, 
                or delete existing groups. Manage your vendor organization.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Open Group Configuration", key="btn_group_config", use_container_width=True):
            st.switch_page("pages/2_⚙️_Group_Configuration.py")
    
    st.markdown("---")
    
    # Quick Stats Section
    st.header("📊 Quick Stats")
    
    # Try to load stats from database
    try:
        import sqlite3
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        DB_PATH = PROJECT_ROOT / "data" / "planner_groups.db"
        
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            
            # Get group stats
            cur.execute("SELECT COUNT(DISTINCT group_name) FROM vendor_group_mapping;")
            num_groups = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(DISTINCT vendor_code) FROM vendor_group_mapping;")
            num_vendors = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM vendor_group_mapping;")
            num_mappings = cur.fetchone()[0]
            
            conn.close()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Groups", num_groups)
            
            with col2:
                st.metric("Total Vendors", num_vendors)
            
            with col3:
                st.metric("Total Mappings", num_mappings)
        else:
            st.info("Database not initialized. Run the seed script first.")
            
    except Exception as e:
        st.warning(f"Could not load stats: {e}")
    
    # Try to load vendor plan stats
    try:
        from data.vendor_state_loader import list_available_vendors
        vendor_files = list_available_vendors()
        
        st.metric("Vendor Plans Available", len(vendor_files))
        
    except Exception as e:
        pass
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>MBM - Master Buy Management System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

