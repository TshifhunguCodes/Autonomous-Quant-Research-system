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

            if self.replay_ohlc.empty or decisions.empty:
                logger.warning(
                    "No replay data available for the selected timeframe: %s to %s",
                    start_date,
                    end_date,
                )
                return False

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
            elif mode == "REPLAY":
                decisions = self.replay_data.get("decisions", pd.DataFrame())
                if not decisions.empty:
                    # Clamp replay_index to valid range - use last row if out of bounds
                    safe_index = min(replay_index, len(decisions) - 1)
                    if safe_index >= 0:
                        df = decisions.iloc[[safe_index]]

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

            # V3 Scoring Logic
            alpha_score = safe_get(row, "alpha_score", 0)
            flow_score = safe_get(row, "flow_score", 0)

            return {
                "time": safe_get(row, "time"),
                "symbol": self.config.market.symbol,
                "regime": safe_get(row, "market_regime", "UNKNOWN"),
                "state": safe_get(row, "market_state", "UNKNOWN"),
                "session": safe_get(row, "session", "UNKNOWN"),
                "h1_bias": safe_get(row, "h1_bias", "UNKNOWN"),
                "volatility": "HIGH" if safe_get(row, "volatility", 0) == 1 else "NORMAL",
                "alpha_score": alpha_score,
                "flow_score": flow_score,
                "current_price": safe_get(row, "close"),
                "current_zone": f"S:{safe_get(row, 'support_level'):.2f} R:{safe_get(row, 'resistance_level'):.2f}",
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
            # Allow both demo and live accounts
            if acc_info and acc_info.trade_mode in [mt5.ACCOUNT_TRADE_MODE_DEMO, mt5.ACCOUNT_TRADE_MODE_REAL]:
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
            elif mode == "REPLAY":
                if not self.replay_ohlc.empty:
                    # Clamp replay_index to valid range
                    safe_index = min(replay_index, len(self.replay_ohlc) - 1)
                    start_idx = max(0, safe_index - num_candles + 1)
                    df = self.replay_ohlc.iloc[start_idx:safe_index+1]
            
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
                        combined = cleaned.iloc[0].to_dict()
                        total_trades = combined.get("trades", 0)
                        win_rate = combined.get("win_rate", 0)
                        combined["wins"] = int(total_trades * win_rate / 100) if total_trades > 0 else 0
                        combined["losses"] = total_trades - combined["wins"]
                        return {"COMBINED": combined}
            except Exception as e:
                logger.error(f"Error processing BACKTEST summary: {e}")
            return {"COMBINED": {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "status": "N/A"}}
            
        elif mode == "REPLAY":
            # Compute dynamic running stats for replay
            stats = {"pnl": 0.0, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, 
                     "balance": getattr(self.config.backtest, 'starting_balance', 10000)}
            
            if self.replay_ohlc.empty and mode == "REPLAY":
                return {"REPLAY": stats}
            
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
                    stats["wins"] = int((closed['result'] == 'WIN').sum())
                    stats["losses"] = int((closed['result'] == 'LOSE').sum())
                    stats["balance"] = round(self.config.backtest.starting_balance + stats["pnl"], 2)
                    # Calculate win_rate if there are closed trades
                    if stats["trades"] > 0:
                        stats["win_rate"] = round((stats["wins"] / stats["trades"]) * 100, 2)
                    else:
                        stats["win_rate"] = 0.0
                    
                    # Unrealized PnL for Equity
                    active_mask = (trades['signal_time'] <= current_time) & \
                                  ((trades['exit_time'] > current_time) | trades['exit_time'].isna())
                    unrealized = 0
                    for _, t in trades[active_mask].iterrows():
                        side_mult = 1 if t['side'].lower() == 'buy' else -1
                        price_diff = (current_price - t['executed_entry']) * side_mult
                        unrealized += (price_diff / t['risk_distance']) * t['risk_amount'] if t['risk_distance'] != 0 else 0
                    
                    stats["equity"] = round(stats["balance"] + unrealized, 2)
            
            # If no replay trades closed yet, show backtest data as demo
            if stats["trades"] == 0:
                try:
                    summary = pd.read_csv(self.backtest_summary_path)
                    if not summary.empty:
                        cleaned = self._clean_dataframe_for_json(summary)
                        if not cleaned.empty:
                            combined = cleaned.iloc[0]
                            total_trades = combined.get("trades", 0)
                            win_rate = combined.get("win_rate", 0)
                            num_wins = int(total_trades * win_rate / 100) if total_trades > 0 else 0
                            num_losses = total_trades - num_wins
                            stats.update({
                                "pnl": combined.get("pnl", 0),
                                "trades": total_trades,
                                "wins": num_wins,
                                "losses": num_losses,
                                "win_rate": win_rate,
                                "profit_factor": combined.get("profit_factor", 0),
                                "balance": combined.get("ending_balance", self.config.backtest.starting_balance),
                                "equity": combined.get("ending_balance", self.config.backtest.starting_balance)
                            })
                except Exception as e:
                    logger.error(f"Error processing BACKTEST summary for replay demo: {e}")
            
            return {"REPLAY": stats}

        # Standard LIVE logic (Mode == LIVE)
        if not self.audit_path.exists():
            # If no live audit, show backtest summary as demo data
            try:
                summary = pd.read_csv(self.backtest_summary_path)
                if not summary.empty:
                    cleaned = self._clean_dataframe_for_json(summary)
                    if not cleaned.empty:
                        combined = cleaned.iloc[0].to_dict()
                        total_trades = combined.get("trades", 0)
                        win_rate = combined.get("win_rate", 0)
                        num_wins = int(total_trades * win_rate / 100) if total_trades > 0 else 0
                        num_losses = total_trades - num_wins
                        # Map combined to ALPHA and FLOW_EXP for display
                        return {
                            "ALPHA": {"pnl": combined.get("pnl", 0), "trades": total_trades, "wins": num_wins, "losses": num_losses, "blocked": 0, "last_status": "DEMO"}, 
                            "FLOW_EXP": {"pnl": combined.get("pnl", 0), "trades": total_trades, "wins": num_wins, "losses": num_losses, "blocked": 0, "last_status": "DEMO"}
                        }
            except Exception as e:
                logger.error(f"Error processing BACKTEST summary for demo: {e}")
            return {
                "ALPHA": {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "blocked": 0, "last_status": "N/A"}, 
                "FLOW_EXP": {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "blocked": 0, "last_status": "N/A"}
            }

        df = pd.read_csv(self.audit_path, low_memory=False)
        executed = df[df["status"] == "EXECUTED"].copy() if "status" in df.columns else pd.DataFrame()
        
        stats = {}
        for sys in ["ALPHA", "FLOW_EXP"]:
            sys_df = executed[executed["system"] == sys] if not executed.empty else pd.DataFrame()
            # Safely get last_status, checking if status column exists and df is not empty
            last_status = "N/A"
            wins = 0
            losses = 0
            pnl = 0.0
            if not sys_df.empty:
                if "status" in sys_df.columns:
                    last_status = sys_df.iloc[-1]["status"]
                if "pnl" in sys_df.columns:
                    pnl = sys_df["pnl"].sum()
                    wins = int((sys_df["pnl"] > 0).sum())
                    losses = int((sys_df["pnl"] < 0).sum())
                elif "result" in sys_df.columns:
                    wins = int((sys_df["result"] == "WIN").sum())
                    losses = int((sys_df["result"] == "LOSE").sum())
                    pnl = sys_df.get("pnl", pd.Series([0]*len(sys_df))).sum()
            stats[sys] = {
                "trades": len(sys_df),
                "wins": wins,
                "losses": losses,
                "pnl": round(pnl, 2),
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