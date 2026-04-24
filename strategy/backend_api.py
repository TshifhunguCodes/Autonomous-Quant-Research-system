from fastapi import FastAPI
import pandas as pd
from strategy.state_manager import DashboardStateManager
from core.logging_utils import get_logger
import uvicorn
import numpy as np
import subprocess
from datetime import date

logger = get_logger(__name__)

app = FastAPI(title="AQRS Real-Time API")
state_manager = DashboardStateManager()

# Global state for replay mode
replay_state = {
    "running": False,
    "index": 0,
    "total_candles": 0,
    "start_date": None,
    "end_date": None,
    "data_loaded": False
}

@app.get("/state")
def get_state(mode: str = "LIVE", replay_index: int = 0):
    market_state = state_manager.get_market_state(mode=mode, replay_index=replay_index)
    account_info = {}
    if mode == "LIVE":
        account_info = state_manager.get_mt5_account_info()
    return {
        "market": market_state,
        "account": account_info
    }

@app.get("/trades")
def get_trades(mode: str = "LIVE", replay_index: int = 0):
    return state_manager.get_trades(mode=mode, replay_index=replay_index)

@app.get("/performance")
def get_performance(mode: str = "LIVE", replay_index: int = 0):
    return state_manager.get_performance_stats(mode=mode, replay_index=replay_index)

@app.get("/signals")
def get_signals(mode: str = "LIVE", replay_index: int = 0):
    # Returns latest 50 setups for the signal feed
    try:
        if mode == "LIVE":
            df = pd.read_csv(state_manager.setup_path, low_memory=False).tail(50)
        elif mode == "REPLAY" and not state_manager.replay_data.get("decisions", pd.DataFrame()).empty:
            df = state_manager.replay_data["decisions"]
            if replay_index < len(df):
                df = df.iloc[:replay_index + 1].tail(50)
            else: # Replay finished
                df = df.tail(50)
        else:
            return []

        # Replace NaN with None and convert numpy types to Python native types
        df = df.replace({np.nan: None})
        for col in df.select_dtypes(include=[np.integer, np.floating]).columns:
            df[col] = df[col].apply(lambda x: x.item() if pd.notna(x) else None)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return []

@app.get("/chart_data")
def get_chart_data(mode: str = "LIVE", replay_index: int = 0, num_candles: int = 100):
    df = state_manager.get_chart_data(mode=mode, replay_index=replay_index, num_candles=num_candles)
    data_to_return = df.to_dict(orient="records")
    logger.debug(f"Returning chart data (first 5 records): {data_to_return[:5]}")
    return data_to_return

@app.get("/replay_control")
def replay_control(action: str, start_date: str = None, end_date: str = None, index: int = 0):
    global replay_state
    if action == "start":
        replay_state["running"] = True
        replay_state["index"] = 0
        replay_state["start_date"] = date.fromisoformat(start_date)
        replay_state["end_date"] = date.fromisoformat(end_date)
        
        # Trigger main.py --mode replay in a subprocess
        cmd = ["python", "main.py", "--mode", "replay",
               "--replay-start", start_date, "--replay-end", end_date]
        subprocess.Popen(cmd) # Non-blocking call
        
        # Load the generated replay data
        if state_manager._load_replay_data(replay_state["start_date"], replay_state["end_date"]):
            replay_state["total_candles"] = len(state_manager.replay_ohlc)
            replay_state["data_loaded"] = True
        else:
            replay_state["running"] = False # Stop if data load fails
            replay_state["data_loaded"] = False
    elif action == "pause":
        replay_state["running"] = False
    elif action == "reset":
        replay_state["running"] = False
        replay_state["index"] = 0
        replay_state["total_candles"] = 0
        replay_state["data_loaded"] = False
        state_manager.replay_data = {} # Clear loaded data
        state_manager.replay_ohlc = pd.DataFrame()
    elif action == "step":
        replay_state["index"] = index
        if replay_state["index"] >= replay_state["total_candles"]:
            replay_state["running"] = False # Auto-pause at end
    
    return replay_state

@app.get("/backtest_control")
def backtest_control(action: str, start_date: str = None, end_date: str = None):
    if action == "run":
        # Trigger main.py --mode backtest in a subprocess
        cmd = ["python", "main.py", "--mode", "backtest", "--refresh-data",
               "--in-sample-end", start_date, "--oos-start", end_date] # Using IS/OOS for date range
        subprocess.Popen(cmd) # Non-blocking call
        return {"status": "Backtest started", "command": " ".join(cmd)}
    return {"status": "Invalid action"}

if __name__ == "__main__":
    # Run with: python -m dashboard.backend_api
    uvicorn.run(app, host="127.0.0.1", port=8001)