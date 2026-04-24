# AQRS V2 Maintenance & Operations Tools

This folder contains utilities to ensure the system is correctly configured and functioning.

### 1. `verify_env.py`
Checks Python dependencies, data directory structures, and the MetaTrader 5 terminal connection.

### 2. `smoke_test.py`
Executes a minimal research and backtest pass using existing data to ensure the pipeline logic hasn't regressed.

### 3. `start_dashboard.py`
Orchestrates the startup of the FastAPI backend and the Streamlit frontend.

---
*Run these scripts from the root directory or directly from this folder.*