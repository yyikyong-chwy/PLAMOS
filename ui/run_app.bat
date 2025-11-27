@echo off
REM Launch script for Vendor Plan Viewer Streamlit App
REM Run this from the project root directory

echo Starting Vendor Plan Viewer...
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit is not installed.
    echo Please install it with: pip install streamlit
    echo Or with uv: uv add streamlit
    pause
    exit /b 1
)

REM Launch the app
streamlit run ui\vendor_plan_viewer.py

pause

