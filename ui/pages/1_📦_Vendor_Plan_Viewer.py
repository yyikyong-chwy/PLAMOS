"""
Vendor Plan Viewer Page
Redirects to vendor_plan_viewer2.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and run the main function from vendor_plan_viewer2
from ui.vendor_plan_viewer2 import main

main()

