import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from core.config import load_config
import numpy as np
from core.logging_utils import get_logger

logger = get_logger(__name__)

class DashboardStateManager:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        self.audit_path = self.config.paths.backtest_dir.parent / "live" / "execution_audit.csv"
        self.setup_path = self.config.paths.trade_setups
        self.replay_decisions_path = self.config.paths.replay_decisions
        self.replay_events_path = self.config.paths.replay_events
        self.backtest_summary_path = self.config.paths.backtest_summary
        self.clean_m5_path = self.config.paths.clean_m5
        
        self.replay_data = {} 
        self.replay_ohlc = pd.DataFrame()
        self.replay_trades = pd.DataFrame()

    def _load_replay_data(self, start_date, end_date):
        """Loads replay decisions and events for a given date range."""
        try:
            decisions_df = pd.read_csv(self.replay_decisions_path, parse_dates=["time"], low_memory=False)
            m5_df = pd.read_csv(self.clean_m5_path, parse_dates=["time"], low_memory=False)

            # Filter by date range
            self.replay_ohlc = m5_df[(m5_df["time"].dt.date >= start_date) & (m5_df["time"].dt.date <= end_date)].reset_index(drop=True)
            decisions = decisions_df[(decisions_df["time"].dt.date >= start_date) & (decisions_df["time"].dt.date <= end_date)].reset_index(drop=True)
            
            # Load simulated trades if they exist
            if self.config.paths.replay_trades.exists():
                self.replay_trades = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'], low_memory=False)

            self.replay_data = {"decisions": self._clean_dataframe_for_json(decisions)}
            return True
        except Exception as e:
            logger.error(f"Failed to load replay data: {e}")
            return False

    def _clean_dataframe_for_json(self, df):
        """Replaces NaN with None and converts numpy types to Python native types."""
        if df is None or df.empty: # Ensure it's a DataFrame before processing
            return pd.DataFrame()
        
        # Replace Infinity with NaN, then replace all NaN with None for JSON safety
        # Standard Python types (int, float) are preserved, numpy scalars are converted by to_dict
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.where(pd.notnull(df), None)

    def get_market_state(self, mode="LIVE", replay_index=0):
        """Extracts the latest regime and signal data."""
        try:
            df = pd.DataFrame()
            if mode == "LIVE":
                df = pd.read_csv(self.setup_path, low_memory=False).tail(1)
            elif mode == "REPLAY" and not self.replay_data.get("decisions", pd.DataFrame()).empty:
                if replay_index < len(self.replay_data["decisions"]):
                    df = self.replay_data["decisions"].iloc[[replay_index]]

            if df.empty: return {}
            row = df.iloc[0]

            # Helper to safely get values, convert numpy scalars, and replace NaN with None
            def safe_get(series, key, default_val=None):
                val = series.get(key, default_val)
                if pd.isna(val):
                    return None
                if type(val) in [int, float, str, bool]:
                    return val
                if hasattr(val, 'item'):
                    return val.item()
                return val

            alpha_score = safe_get(row, "confirm_score") if safe_get(row, "quality") == "ELITE" else 0
            flow_score = safe_get(row, "confirm_score")

            return {
                "time": safe_get(row, "time"),
                "symbol": self.config.market.symbol,
                "regime": safe_get(row, "market_regime", "UNKNOWN"),
                "state": safe_get(row, "market_state", "UNKNOWN"),
                "session": safe_get(row, "session", "UNKNOWN"),
                "h1_bias": safe_get(row, "h1_bias", "UNKNOWN"),
                "volatility": safe_get(row, "market_state", "UNKNOWN"),
                "alpha_score": alpha_score,
                "flow_score": flow_score,
                "current_price": safe_get(row, "close"),
                "current_zone": f"S:{safe_get(row, 'near_support')} R:{safe_get(row, 'near_resistance')}",
                "setup": safe_get(row, "setup", "NONE"),
                "confirmed_signal": safe_get(row, "confirmed_signal", "none").upper()
            }
        except Exception:
            return {}

    def get_trades(self, mode="LIVE", replay_index=0):
        """Fetches active positions from MT5 and history from audit log."""
        active_positions = []
        if mode == "LIVE" and mt5.initialize():
            acc_info = mt5.account_info()
            # Safety Check: Hard block if not Demo
            if acc_info and acc_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO:
                positions = mt5.positions_get(symbol=self.config.market.symbol)
                if positions:
                    for p in positions:
                        active_positions.append({
                            "ticket": p.ticket, # int
                            "type": "BUY" if p.type == 0 else "SELL",
                            "volume": p.volume, # float
                            "price_open": p.price_open, # float
                            "sl": p.sl,
                            "tp": p.tp,
                            "pnl": p.profit, # float
                            "comment": p.comment
                        })
        elif mode == "REPLAY" and not self.replay_ohlc.empty:
            idx = min(replay_index, len(self.replay_ohlc) - 1)
            # Reconstruct simulated active positions at this specific point in time
            current_candle = self.replay_ohlc.iloc[idx]
            current_time = current_candle['time']
            current_price = current_candle['close']
            
            # Use memory-cached trades if possible, else read disk
            trades = self.replay_trades if not self.replay_trades.empty else None
            if trades is None and self.config.paths.replay_trades.exists():
                trades = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'], low_memory=False)
            
            if trades is not None:
                try:
                    active_mask = (trades['signal_time'] <= current_time) & \
                                  ((trades['exit_time'] > current_time) | trades['exit_time'].isna())
                    
                    for _, t in trades[active_mask].iterrows():
                        side_mult = 1 if t['side'] == 'buy' else -1
                        price_diff = (current_price - t['executed_entry']) * side_mult
                        unrealized_pnl = (price_diff / t['risk_distance']) * t['risk_amount'] if t['risk_distance'] != 0 else 0
                        
                        active_positions.append({
                            "ticket": "Simulated",
                            "type": t['side'].upper(),
                            "volume": round(float(t.get('risk_multiplier', 1.0)), 2),
                            "price_open": t['executed_entry'],
                            "sl": t['stop_loss'],
                            "tp": t['take_profit'],
                            "pnl": round(unrealized_pnl, 2),
                            "comment": t.get('setup', 'REPLAY')
                        })
                except Exception as e:
                    logger.error(f"Error reconstructing active replay trades: {e}")
        
        history = pd.DataFrame()
        if mode == "LIVE" and self.audit_path.exists():
            try:
                history = pd.read_csv(self.audit_path).tail(50)
                if not history.empty:
                    history = self._clean_dataframe_for_json(history)
            except Exception as e:
                logger.error(f"Error reading LIVE history: {e}")
                history = pd.DataFrame()
        elif mode == "REPLAY" and self.config.paths.replay_events.exists():
            try:
                # Filter events to match the current replay progress
                idx = min(replay_index, len(self.replay_ohlc) - 1) if not self.replay_ohlc.empty else 0
                current_time = self.replay_ohlc.iloc[idx]['time'] if not self.replay_ohlc.empty else None
                
                history = pd.read_csv(self.config.paths.replay_events, parse_dates=['time'])
                if current_time:
                    history = history[history['time'] <= current_time].tail(50)
                
                if not history.empty:
                    # Normalize columns for the dashboard's chart and feed logic
                    history = history.rename(columns={
                        "event": "status",
                        "time": "signal_time",
                        "decision": "retcode",
                        "trade_id": "ticket"
                    })
                    # Map internal events to dashboard-recognized statuses
                    history["status"] = history["status"].replace("TRADE_OPENED", "EXECUTED")
                    history["time"] = history["signal_time"] # Ensure dual time columns
                    history = self._clean_dataframe_for_json(history)
            except Exception as e:
                logger.error(f"Error reading REPLAY history: {e}")
                history = pd.DataFrame()

        return {
            "active": active_positions,
            "history": history.to_dict(orient="records") if not history.empty else []
        }

    def get_chart_data(self, mode="LIVE", replay_index=0, num_candles=100):
        """Provides OHLC data for charting."""
        df = pd.DataFrame()
        try:
            if mode == "LIVE":
                df = pd.read_csv(self.config.paths.clean_m5, parse_dates=["time"], low_memory=False).tail(num_candles)
            elif mode == "REPLAY" and not self.replay_ohlc.empty:
                df = self.replay_ohlc.iloc[max(0, replay_index-num_candles+1):replay_index+1]
            
            if not df.empty:
                df = self._clean_dataframe_for_json(df)
        except Exception as e:
            logger.error(f"Error getting chart data: {e}")
            # df remains an empty DataFrame on error
        return df # Always return a DataFrame

    def get_performance_stats(self, mode="LIVE", replay_index=0):
        """Computes Alpha vs Flow metrics from the audit log."""
        if mode == "BACKTEST":
            try:
                summary = pd.read_csv(self.backtest_summary_path)
                if not summary.empty:
                    cleaned = self._clean_dataframe_for_json(summary)
                    if not cleaned.empty:
                        return {"COMBINED": cleaned.iloc[0].to_dict()}
            except Exception as e:
                logger.error(f"Error processing BACKTEST summary: {e}")
            return {"COMBINED": {"pnl": 0, "trades": 0, "status": "N/A"}}
            
        elif mode == "REPLAY":
            # Compute dynamic running stats for replay
            stats = {"pnl": 0.0, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, 
                     "balance": self.config.backtest.starting_balance, "equity": self.config.backtest.starting_balance}
            
            if not self.replay_ohlc.empty:
                idx = min(replay_index, len(self.replay_ohlc) - 1)
                current_time = self.replay_ohlc.iloc[idx]['time']
                current_price = self.replay_ohlc.iloc[idx]['close']
                
                trades = self.replay_trades if not self.replay_trades.empty else None
                if trades is None and self.config.paths.replay_trades.exists():
                    trades = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'])
                
                if trades is not None and not trades.empty:
                    # Realized PnL (trades closed at or before current candle)
                    closed = trades[trades['exit_time'] <= current_time]
                    stats["pnl"] = round(float(closed['pnl'].sum()), 2)
                    stats["trades"] = len(closed)
                    stats["balance"] = round(self.config.backtest.starting_balance + stats["pnl"], 2)
                    
                    if len(closed) > 0:
                        wins = (closed['result'] == 'WIN').sum()
                        stats["win_rate"] = round((wins / len(closed)) * 100, 2)
                    
                    # Unrealized PnL for Equity
                    active_mask = (trades['signal_time'] <= current_time) & \
                                  ((trades['exit_time'] > current_time) | trades['exit_time'].isna())
                    unrealized = 0
                    for _, t in trades[active_mask].iterrows():
                        side_mult = 1 if t['side'].lower() == 'buy' else -1
                        price_diff = (current_price - t['executed_entry']) * side_mult
                        unrealized += (price_diff / t['risk_distance']) * t['risk_amount'] if t['risk_distance'] != 0 else 0
                    
                    stats["equity"] = round(stats["balance"] + unrealized, 2)
            
            return {"REPLAY": stats}

        # Standard LIVE logic (Mode == LIVE)
        if not self.audit_path.exists():
            return {
                "ALPHA": {"pnl": 0, "trades": 0, "blocked": 0, "last_status": "N/A"}, 
                "FLOW_EXP": {"pnl": 0, "trades": 0, "blocked": 0, "last_status": "N/A"}
            }

        df = pd.read_csv(self.audit_path, low_memory=False)
        executed = df[df["status"] == "EXECUTED"].copy() if "status" in df.columns else pd.DataFrame()
        
        stats = {}
        for sys in ["ALPHA", "FLOW_EXP"]:
            sys_df = executed[executed["system"] == sys] if not executed.empty else pd.DataFrame()
            # Safely get last_status, checking if status column exists and df is not empty
            last_status = "N/A"
            if not sys_df.empty and "status" in sys_df.columns:
                last_status = sys_df.iloc[-1]["status"]
            stats[sys] = {
                "trades": len(sys_df),
                "blocked": len(df[(df["system"] == sys) & (df["status"] == "BLOCKED")]) if "status" in df.columns else 0,
                "last_status": last_status
            }
        return stats

    def get_mt5_account_info(self):
        if not mt5.initialize():
            return {"connected": False}
        acc = mt5.account_info()
        if not acc: return {"connected": False}

        # Convert numpy floats to Python floats
        balance = acc.balance.item() if isinstance(acc.balance, np.floating) else acc.balance
        equity = acc.equity.item() if isinstance(acc.equity, np.floating) else acc.equity
        return {
            "connected": True,
            "login": acc.login,
            "balance": balance,
            "equity": equity,
            "is_demo": acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        }