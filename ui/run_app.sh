#!/bin/bash
# Launch script for Vendor Plan Viewer Streamlit App (Unix/Mac)
# Run this from the project root directory

echo "Starting Vendor Plan Viewer..."
echo ""

# Check if streamlit is installed
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Streamlit is not installed."
    echo "Please install it with: pip install streamlit"
    echo "Or with uv: uv add streamlit"
    exit 1
fi

# Launch the app
streamlit run ui/vendor_plan_viewer.py

