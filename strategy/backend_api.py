from fastapi import FastAPI
import sys
import pandas as pd
from strategy.state_manager import DashboardStateManager
from core.logging_utils import get_logger
import uvicorn
import numpy as np
import subprocess
from datetime import date, datetime

logger = get_logger(__name__)

app = FastAPI(title="AQRS Real-Time API")
state_manager = DashboardStateManager()
replay_generation_process = None # Global to hold the Popen object

def _clean_for_json(data):
    """Recursive helper to make data JSON-serializable (handles NumPy, Timestamps, NaN)."""
    if data is None:
        return None

    # Handle collections first to avoid pd.isna() ValueError on lists/dicts
    if isinstance(data, list):
        return [_clean_for_json(i) for i in data]
    if isinstance(data, dict):
        return {k: _clean_for_json(v) for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return [_clean_for_json(i) for i in data.tolist()]

    # Handle pandas/numpy Nulls
    if isinstance(data, (float, np.floating)) and np.isnan(data):
        return None
    if data is pd.NA:
        return None

    if isinstance(data, (datetime, date, pd.Timestamp)):
        return data.isoformat()
    if isinstance(data, (np.generic, np.integer, np.floating)):
        return data.item()
    return data

@app.get("/health")
def health_check():
    """Simple endpoint to verify API is online."""
    return {"status": "online", "timestamp": datetime.now().isoformat()}


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
    
    # Ensure we always return valid data structure for REPLAY mode
    if mode == "REPLAY" and not market_state:
        logger.warning(f"Empty market state for REPLAY mode at index {replay_index}. Total decisions: {len(state_manager.replay_data.get('decisions', []))}, Total ohlc: {len(state_manager.replay_ohlc)}")
        market_state = {}
    
    return _clean_for_json({
        "market": market_state,
        "account": account_info
    })

@app.get("/trades")
def get_trades(mode: str = "LIVE", replay_index: int = 0):
    trades_data = state_manager.get_trades(mode=mode, replay_index=replay_index)
    return _clean_for_json(trades_data)

@app.get("/performance")
def get_performance(mode: str = "LIVE", replay_index: int = 0):
    return _clean_for_json(state_manager.get_performance_stats(mode=mode, replay_index=replay_index))

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

        return _clean_for_json(df.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return []

@app.get("/chart_data")
def get_chart_data(mode: str = "LIVE", replay_index: int = 0, num_candles: int = 100):
    df = state_manager.get_chart_data(mode=mode, replay_index=replay_index, num_candles=num_candles)
    data_to_return = _clean_for_json(df.to_dict(orient="records"))
    logger.debug(f"Returning chart data (first 5 records): {data_to_return[:5]}")
    return data_to_return

@app.get("/replay_control")
def replay_control(action: str, start_date: str = None, end_date: str = None, index: int = 0):
    global replay_state
    global replay_generation_process
    if action == "status":
        return replay_state

    if action == "start":
        if not start_date or not end_date:
            return {"error": "Missing replay start/end dates", "total_candles": 0, "data_loaded": False}
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD.", "total_candles": 0, "data_loaded": False}
        if start_dt > end_dt:
            return {"error": "Replay start date must be on or before end date.", "total_candles": 0, "data_loaded": False}

        # Reset replay state for a new generation
        replay_state["running"] = False # Not running playback yet, just generation
        replay_state["index"] = 0
        replay_state["start_date"] = start_dt
        replay_state["end_date"] = end_dt
        replay_state["data_loaded"] = False
        replay_state["total_candles"] = 0
        
        if replay_generation_process and replay_generation_process.poll() is None:
            logger.warning("Replay generation already in progress. Ignoring new request.")
            return {"status": "already_started"}

        # V3 Migration: Use main_v3.py for replay generation
        cmd = [sys.executable, "main_v3.py", "--mode", "replay",
               "--replay-start", start_date, "--replay-end", end_date, "--output", "data/replay/replay_decisions.csv", "--skip-readiness"]

        try:
            replay_generation_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Started replay generation in background for %s to %s", start_date, end_date)
            return {"status": "generation_started"}
        except Exception as e:
            logger.error(f"Failed to launch replay generation: {e}")
            return {"status": "error", "message": str(e)}

    elif action == "check_generation_status":
        if replay_generation_process is None:
            return {"status": "not_started"}
        
        if replay_generation_process.poll() is None: # Process is still running
            return {"status": "in_progress"}
        else: # Process has finished
            logger.info("Replay generation process finished. Return code: %s", replay_generation_process.returncode)
            if replay_generation_process.returncode != 0:
                logger.error("Replay generation failed with return code %s", replay_generation_process.returncode)
                replay_generation_process = None
                return {"status": "failed", "returncode": replay_generation_process.returncode}
            
            # Attempt to load data after generation
            success = state_manager._load_replay_data(replay_state["start_date"], replay_state["end_date"])
            if success:
                replay_state["total_candles"] = len(state_manager.replay_ohlc)
                replay_state["data_loaded"] = True
                replay_state["running"] = True # Ready for playback
                logger.info("Replay data loaded successfully. Total candles: %s", replay_state["total_candles"])
                replay_generation_process = None # Clear the process
                return {"status": "completed", "total_candles": replay_state["total_candles"]}
            else:
                logger.error("Failed to load replay data after generation.")
                replay_state["data_loaded"] = False
                replay_state["running"] = False
                replay_generation_process = None
                return {"status": "failed_to_load_data"}

    elif action == "pause":
        replay_state["running"] = False
    elif action == "reset":
        replay_state["running"] = False
        replay_state["index"] = 0
        replay_state["total_candles"] = 0
        replay_state["data_loaded"] = False
        state_manager.replay_data = {} # Clear loaded data
        state_manager.replay_ohlc = pd.DataFrame()
        if replay_generation_process:
            replay_generation_process.terminate()
            replay_generation_process = None
    elif action == "step":
        replay_state["index"] = index
        if replay_state["index"] >= replay_state["total_candles"]:
            replay_state["running"] = False # Auto-pause at end
    
    return replay_state

@app.get("/backtest_control")
def backtest_control(action: str, start_date: str = None, end_date: str = None):
    if action == "run":
        # V3 Migration: Use main_v3.py for institutional backtests
        cmd = [sys.executable, "main_v3.py", "--mode", "backtest", 
               "--output", "data/backtest/v3_research_output.csv"]
        subprocess.Popen(cmd) # Non-blocking call
        return {"status": "Backtest started", "command": " ".join(cmd)}
    return {"status": "Invalid action"}

@app.get("/intelligence")
def get_intelligence(mode: str = "LIVE", replay_index: int = 0):
    return _clean_for_json(state_manager.get_expectancy_intelligence(mode=mode, replay_index=replay_index))

@app.get("/signals")
def get_signals_api(mode: str = "LIVE", replay_index: int = 0):
    # This endpoint is now handled by the state_manager directly in the backend
    # and will return the filtered signals.
    return _clean_for_json(state_manager.get_signals(mode=mode, replay_index=replay_index))

if __name__ == "__main__":
    # Run with: python -m dashboard.backend_api
    uvicorn.run(app, host="127.0.0.1", port=8001)