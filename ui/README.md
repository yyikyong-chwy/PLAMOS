# Vendor Container Plan Viewer

A Streamlit web application for viewing and analyzing vendor container plans.

## Features

- 📊 **Load Vendor States**: Automatically loads all vendor states from `data/vendor_plans/` folder
- 🏢 **Vendor Selection**: Choose from available vendors with easy-to-use dropdown
- 📋 **Multiple Plans View**: View and compare all plans for a selected vendor
- 📈 **Metrics Dashboard**: Comprehensive metrics display including:
  - Overall Score
  - Container Utilization
  - Accuracy Metrics (APE vs Planned/Base/Excess)
  - Violations and Issues
  - Container-specific breakdowns
- 🔄 **Plan Comparison**: Compare multiple plans with:
  - Bar charts for key metrics
  - Radar charts for multi-dimensional comparison
  - Trade-offs analysis
- 🔍 **Detailed Plan View**: Filter and search container assignments by:
  - Destination
  - Container ID
  - SKU number
- 📊 **Visualizations**: Interactive charts showing CBM distribution and utilization

## Installation

1. Make sure you have the required dependencies installed:

```bash
pip install streamlit pandas plotly
```

Or if using uv:

```bash
uv add streamlit pandas plotly
```

## Running the App

From the project root directory, run:

```bash
streamlit run ui/vendor_plan_viewer2.py
```

Or if using uv:

```bash
uv run streamlit run ui/vendor_plan_viewer2.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Load Vendors**: The app automatically loads all vendors from `data/vendor_plans/` on startup
2. **Select a Vendor**: Use the sidebar dropdown to choose a vendor
3. **View Metrics**: See comprehensive metrics for each plan
4. **Compare Plans**: If multiple plans exist, view comparison charts and trade-offs
5. **Drill Down**: Click on individual plan tabs to see detailed container assignments
6. **Filter Data**: Use filters to narrow down to specific destinations, containers, or SKUs

## Data Structure

The app expects vendor state data to be stored in:
```
data/vendor_plans/
  ├── {vendor_code_1}/
  │   └── vendor_state.json
  ├── {vendor_code_2}/
  │   └── vendor_state.json
  └── ...
```

Each `vendor_state.json` should contain a serialized `vendorState` object with:
- Vendor information
- List of container plans
- Metrics for each plan
- Container assignments

## Metrics Explained

- **Overall Score**: Composite score (0-100) indicating plan quality
- **Avg Utilization**: Average container space utilization percentage
- **APE vs Planned**: Average Percentage Error compared to planned quantities
- **APE vs Base**: Average Percentage Error compared to base demand
- **Low Util Count**: Number of containers below utilization threshold (default 90%)
- **MOQ Violations**: Number of Minimum Order Quantity violations
- **SKU Splits Off MCP**: SKU splits not aligned with Master Case Pack

## Tips

- Use the **Reload** button in the sidebar to refresh data after changes
- The comparison view helps identify the best-performing plan
- Filter options make it easy to focus on specific containers or destinations
- Export data using the download button on data tables

## Troubleshooting

If you encounter issues:

1. **No vendors found**: Ensure `data/vendor_plans/` exists and contains vendor subdirectories
2. **Import errors**: Make sure you're running from the project root directory
3. **Data loading errors**: Check that vendor_state.json files are valid JSON

## Development

The app is built with:
- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **Pydantic**: Data validation (via vendorState models)

