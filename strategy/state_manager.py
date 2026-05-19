import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from core.config import load_config
import numpy as np
from core.logging_utils import get_logger

logger = get_logger(__name__)

def get_csv_tail(path: Path, n: int = 1) -> pd.DataFrame:
    """Efficiently reads the last n lines of a CSV file without loading the whole thing."""
    if not path.exists():
        return pd.DataFrame()
    try:
        # Count lines without loading full dataframe columns/data
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for _ in f)
        return pd.read_csv(path, skiprows=range(1, max(1, count - n)), on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, on_bad_lines="skip").tail(n) # Fallback

class DashboardStateManager:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        
        # Safe path resolution
        backtest_dir = self.config.paths.backtest_dir if hasattr(self.config.paths, "backtest_dir") else Path("data/backtest")
        
        self.audit_path = backtest_dir.parent / "live" / "execution_audit.csv"
        self.outcomes_path = backtest_dir.parent / "live" / "trade_outcomes.csv"
        self.setup_path = self.config.paths.trade_setups
        self.replay_decisions_path = self.config.paths.replay_decisions
        self.backtest_summary_path = self.config.paths.backtest_summary
        self.clean_m5_path = self.config.paths.clean_m5
        
        self.replay_data = {} 
        self.replay_ohlc = pd.DataFrame()
        self.replay_trades = pd.DataFrame()

    def get_data_bounds(self):
        """Returns the start and end dates of the available historical data."""
        try:
            if self.clean_m5_path.exists():
                # Efficiently get first and last timestamps
                first_row = pd.read_csv(self.clean_m5_path, nrows=1, parse_dates=["time"])
                last_row = get_csv_tail(self.clean_m5_path, 1)
                if not first_row.empty and not last_row.empty:
                    return {
                        "start": first_row.iloc[0]["time"].date(),
                        "end": pd.to_datetime(last_row.iloc[0]["time"]).date()
                    }
        except Exception as e:
            logger.error(f"Error getting data bounds: {e}")
        return {"start": None, "end": None}

    def _load_replay_data(self, start_date, end_date):
        """Loads replay decisions and events for a given date range."""
        import pandas.errors
        try:
            if not self.replay_decisions_path.exists() or self.replay_decisions_path.stat().st_size == 0:
                logger.warning("Replay decisions file is empty or missing.")
                return False

            decisions_df = pd.read_csv(self.replay_decisions_path, parse_dates=["time"], low_memory=False)
            
            # V3 Optimization: Replay decisions file now contains the full 83-column pipeline.
            # Using this as the source for ohlc ensures the dashboard index matches the logic index perfectly.
            if not decisions_df.empty:
                # Defensive check: if decisions_df is missing OHLC data, merge with source to prevent KeyErrors
                if 'close' not in decisions_df.columns and 'time' in decisions_df.columns:
                    logger.warning("Replay decisions missing 'close' column. Attempting recovery from source M5 data.")
                    m5_df = pd.read_csv(self.clean_m5_path, parse_dates=["time"], low_memory=False)
                    decisions_df = pd.merge(decisions_df, m5_df[['time', 'open', 'high', 'low', 'close']], on='time', how='left')
                elif 'close' not in decisions_df.columns:
                    logger.error("Replay artifact is invalid (missing columns). Please restart replay.")
                    return False
                
                self.replay_ohlc = decisions_df.copy()
            else:
                # Fallback to global clean data if decisions are missing
                m5_df = pd.read_csv(self.clean_m5_path, parse_dates=["time"], low_memory=False)
                self.replay_ohlc = m5_df[(m5_df["time"].dt.date >= start_date) & (m5_df["time"].dt.date <= end_date)].reset_index(drop=True)
            
            # Load simulated trades if they exist
            if self.config.paths.replay_trades.exists():
                trades_full = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'], low_memory=False)
                self.replay_trades = trades_full[(trades_full["signal_time"].dt.date >= start_date) & (trades_full["signal_time"].dt.date <= end_date)].reset_index(drop=True)

            # Load events and filter
            if self.config.paths.replay_events.exists():
                events_full = pd.read_csv(self.config.paths.replay_events, parse_dates=['time'], low_memory=False)
                self.replay_events = events_full[(events_full["time"].dt.date >= start_date) & (events_full["time"].dt.date <= end_date)].reset_index(drop=True)
            else:
                self.replay_events = pd.DataFrame()

            if self.replay_ohlc.empty:
                logger.warning(
                    "No OHLC data available for the selected timeframe: %s to %s",
                    start_date,
                    end_date,
                )
                return False

            self.replay_data = {"decisions": self._clean_dataframe_for_json(self.replay_ohlc)}
            return True
        except pd.errors.EmptyDataError:
            logger.error("No data found in replay files for the selected range.")
            return False
        except Exception as e:
            logger.error(f"Failed to load replay data: {e}")
            return False

    def _clean_dataframe_for_json(self, df):
        """Replaces NaN with None and converts numpy types to Python native types."""
        if df is None or not isinstance(df, (pd.DataFrame, pd.Series)) or df.empty:
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
                df = get_csv_tail(self.setup_path, 1)
            elif mode == "REPLAY":
                # V3 Replay: ohlc contains the full 83-column pipeline
                if not self.replay_ohlc.empty:
                    safe_index = min(replay_index, len(self.replay_ohlc) - 1)
                    if safe_index >= 0:
                        df = self.replay_ohlc.iloc[[safe_index]]
            if df.empty: return {}
            row = df.iloc[0]

            # Helper to safely get values, convert numpy scalars, and replace NaN with None
            def safe_get(series, key, default_val=None):
                val = series.get(key, default_val)
                if hasattr(val, 'item'):
                    return val.item()
                if pd.isna(val):
                    return None
                if isinstance(val, (int, float, str, bool)):
                    return val
                return val

            # V3 Scoring Logic
            alpha_score = safe_get(row, "alpha_score", 0)
            flow_score = safe_get(row, "flow_score", 0)

            return {
                "time": safe_get(row, "time"),
                "symbol": self.config.market.symbol,
                "regime": safe_get(row, "market_regime", "NEUTRAL"),
                "behavior": safe_get(row, "behavior_label", "UNKNOWN"),
                "structure_state": safe_get(row, "structure_state", "UNKNOWN"),
                "pattern": safe_get(row, "pattern", "NONE"),
                "state": safe_get(row, "market_state", "RANGING"),
                "session": safe_get(row, "session", "UNKNOWN"),
                "h1_bias": safe_get(row, "h1_bias", "UNKNOWN"),
                "volatility": "HIGH" if safe_get(row, "volatility", 0) == 1 else "NORMAL",
                "alpha_score": alpha_score,
                "flow_score": flow_score,
                "current_price": safe_get(row, "close"),
                "current_zone": f"S:{safe_get(row, 'support_level', 0):.2f} R:{safe_get(row, 'resistance_level', 0):.2f}"
                                if safe_get(row, 'support_level') is not None else "N/A",
                "setup": safe_get(row, "setup", "NONE"),
                "confirmed_signal": safe_get(row, "confirmed_signal", "none").upper(),
                "execution_reason": safe_get(row, "execution_reason", "N/A")
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
            current_time = current_candle.get('time')
            current_price = current_candle.get('close', 0.0)
            
            # Use memory-cached trades if possible, else read disk
            trades = self.replay_trades if not self.replay_trades.empty else None
            if trades is None and self.config.paths.replay_trades.exists():
                trades = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'], low_memory=False)
            
            if trades is not None:
                try:
                    active_mask = (trades['signal_time'] <= current_time) & \
                                  ((trades['exit_time'] > current_time) | trades['exit_time'].isna())
                    
                    for _, t in trades[active_mask].iterrows():
                        # Reconstruct simulated PnL based on current price
                        side_mult = 1 if t['side'] == 'buy' else -1
                        price_diff = (current_price - t['executed_entry']) * side_mult
                        unrealized_pnl = (price_diff / t['risk_distance']) * t['risk_amount'] if t['risk_distance'] != 0 else 0
                        
                        active_positions.append({
                            "ticket": f"SIM_{t.get('trade_id', '0')}",
                            "type": t['side'].upper(),
                            "side": t['side'].lower(),
                            "volume": round(float(t.get('risk_multiplier', 1.0)), 2),
                            "price": t['executed_entry'], # UI looks for 'price' in charts
                            "sl": t['stop_loss'],
                            "tp": t['take_profit'],
                            "pnl": round(unrealized_pnl, 2),
                            "comment": f"{t.get('system', 'REPLAY').upper()} | {t.get('setup', 'N/A')}"
                        })
                except Exception as e:
                    logger.error(f"Error reconstructing active replay trades: {e}")
        
        history = pd.DataFrame()
        if mode == "LIVE" and self.audit_path.exists():
            try:
                history = pd.read_csv(self.audit_path, on_bad_lines="skip").tail(50)
                if not history.empty:
                    history = self._clean_dataframe_for_json(history)
            except Exception as e:
                logger.error(f"Error reading LIVE history: {e}")
                history = pd.DataFrame()
        elif mode == "REPLAY" and not self.replay_ohlc.empty:
            try:
                # Filter events to match the current replay progress
                idx = min(replay_index, len(self.replay_ohlc) - 1)
                current_time = self.replay_ohlc.iloc[idx]['time']
                
                # Use in-memory filtered events
                history_raw = getattr(self, 'replay_events', pd.DataFrame())
                if history_raw.empty and self.config.paths.replay_events.exists():
                    history_raw = pd.read_csv(self.config.paths.replay_events, parse_dates=['time'])
                
                if not history_raw.empty:
                    history = history_raw[history_raw['time'] <= current_time].tail(50).copy()
                
                if not history.empty:
                    # Normalize columns for the dashboard's unified feed logic
                    history = history.rename(columns={
                        "event": "status",
                        "decision": "retcode",
                        "trade_id": "ticket"
                    })
                    # Ensure required columns for dashboard table are present
                    if "system" not in history.columns: history["system"] = "REPLAY"
                    if "side" not in history.columns: history["side"] = "N/A"
                    if "price" not in history.columns: history["price"] = 0.0

                    # Map internal events to dashboard-recognized statuses
                    history["status"] = history["status"].replace("TRADE_OPENED", "EXECUTED")
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
                     "balance": float(getattr(self.config.backtest, 'starting_balance', 1000.0)),
                     "equity": float(getattr(self.config.backtest, 'starting_balance', 1000.0)),
                     "alpha_signals": 0, "flow_signals": 0,
                     "wins": 0, "losses": 0}
            
            if self.replay_ohlc.empty and mode == "REPLAY":
                return {"REPLAY": stats}
            
            if not self.replay_ohlc.empty:
                idx = min(replay_index, len(self.replay_ohlc) - 1)
                candle = self.replay_ohlc.iloc[idx]
                current_time = candle.get('time')
                current_price = candle.get('close', 0.0)

                # Fetch decisions for signal counting
                decisions = pd.DataFrame()
                if isinstance(self.replay_data.get("decisions"), list):
                    decisions = pd.DataFrame(self.replay_data["decisions"])
                    decisions['time'] = pd.to_datetime(decisions['time'])
                elif isinstance(self.replay_data.get("decisions"), pd.DataFrame):
                    decisions = self.replay_data["decisions"]

                if not decisions.empty:
                    # Count signals seen UP TO current time
                    visible_decisions = decisions[decisions['time'] <= current_time]
                    stats["alpha_signals"] = len(visible_decisions[visible_decisions['signal'] == 'ALPHA'])
                    stats["flow_signals"] = len(visible_decisions[visible_decisions['signal'] == 'FLOW'])

                trades = self.replay_trades if not self.replay_trades.empty else None
                if trades is None and self.config.paths.replay_trades.exists():
                    trades = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'])
                
                closed = pd.DataFrame()
                if trades is not None and not trades.empty:
                    # A trade is considered "closed" in performance stats if its exit_time is <= current_time
                    # and it's not marked as "OPEN" (which only happens at the very end of the file)
                    closed = trades[(trades['exit_time'] <= current_time) & (trades['result'] != "OPEN")]
                    
                    stats["pnl"] = round(float(closed['pnl'].sum()), 2)
                    stats["trades"] = len(closed)
                    stats["wins"] = len(closed[closed['pnl'] > 0])
                    stats["losses"] = len(closed[closed['pnl'] < 0])
                    stats["balance"] = round(self.config.backtest.starting_balance + stats["pnl"], 2)
                    
                    if stats["trades"] > 0:
                        stats["win_rate"] = round((stats["wins"] / stats["trades"]) * 100, 2)
                        stats["profit_factor"] = round(abs(closed[closed['pnl'] > 0]['pnl'].sum() / closed[closed['pnl'] < 0]['pnl'].sum()), 2) if stats["losses"] > 0 else 0.0
                    
                    # Unrealized PnL for Equity
                    active_mask = (trades['signal_time'] <= current_time) & \
                                  ((trades['exit_time'] > current_time) | trades['exit_time'].isna())
                    unrealized = 0
                    for _, t in trades[active_mask].iterrows():
                        side_mult = 1 if t['side'].lower() == 'buy' else -1
                        price_diff = (current_price - t['executed_entry']) * side_mult
                        unrealized += (price_diff / t['risk_distance']) * t['risk_amount'] if t['risk_distance'] != 0 else 0
                    
                    stats["equity"] = round(stats["balance"] + unrealized, 2)
            
                # Calculate per-engine splits for the Dual Engine panel
                alpha_trades = closed[closed['system'] == 'ALPHA'] if not closed.empty else pd.DataFrame()
                flow_trades = closed[closed['system'] == 'FLOW_EXP'] if not closed.empty else pd.DataFrame()
            
            return {
                "REPLAY": stats,
                "ALPHA": { 
                    "pnl": alpha_trades['pnl'].sum() if not alpha_trades.empty else 0,
                    "trades": len(alpha_trades),
                    "wins": len(alpha_trades[alpha_trades['pnl'] > 0]) if not alpha_trades.empty else 0,
                    "losses": len(alpha_trades[alpha_trades['pnl'] < 0]) if not alpha_trades.empty else 0,
                    "blocked": 0, "last_status": "REPLAY"
                },
                "FLOW_EXP": {
                    "pnl": flow_trades['pnl'].sum() if not flow_trades.empty else 0,
                    "trades": len(flow_trades),
                    "wins": len(flow_trades[flow_trades['pnl'] > 0]) if not flow_trades.empty else 0,
                    "losses": len(flow_trades[flow_trades['pnl'] < 0]) if not flow_trades.empty else 0,
                    "blocked": 0, "last_status": "REPLAY"
                }
            }

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

        if self.outcomes_path.exists():
            try:
                outcomes = pd.read_csv(self.outcomes_path, low_memory=False, on_bad_lines="skip")
                audit = pd.read_csv(self.audit_path, low_memory=False, on_bad_lines="skip")
                stats = {}
                for sys in ["ALPHA", "FLOW_EXP"]:
                    sys_df = outcomes[outcomes["system"] == sys] if "system" in outcomes.columns else pd.DataFrame()
                    blocked = len(audit[(audit["system"] == sys) & (audit["status"] == "BLOCKED")]) if {"system", "status"}.issubset(audit.columns) else 0
                    stats[sys] = {
                        "trades": len(sys_df),
                        "wins": int((sys_df["pnl"] > 0).sum()) if "pnl" in sys_df.columns else 0,
                        "losses": int((sys_df["pnl"] < 0).sum()) if "pnl" in sys_df.columns else 0,
                        "pnl": round(float(sys_df["pnl"].sum()), 2) if "pnl" in sys_df.columns else 0.0,
                        "blocked": blocked,
                        "last_status": "OUTCOMES_SYNCED" if not sys_df.empty else "NO_CLOSED_TRADES",
                    }
                return stats
            except Exception as e:
                logger.error("Error reading LIVE outcome performance data: %s", e)

        try:
            df = pd.read_csv(self.audit_path, low_memory=False, on_bad_lines="skip")
        except Exception as e:
            logger.error("Error reading LIVE audit performance data: %s", e)
            return {
                "ALPHA": {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "blocked": 0, "last_status": "AUDIT_READ_ERROR"},
                "FLOW_EXP": {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "blocked": 0, "last_status": "AUDIT_READ_ERROR"},
            }
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

    def get_expectancy_intelligence(self, mode="LIVE", replay_index=0):
        """Calculates the expectancy matrix for the 'What Works Now' intelligence tab."""
        # For Replay, we use the simulated outcomes up to current index
        if mode == "REPLAY":
            if self.replay_ohlc.empty: return None
            idx = min(replay_index, len(self.replay_ohlc) - 1)
            current_time = self.replay_ohlc.iloc[idx]['time']
            
            if self.replay_trades.empty and self.config.paths.replay_trades.exists():
                self.replay_trades = pd.read_csv(self.config.paths.replay_trades, parse_dates=['signal_time', 'exit_time'])
            
            if self.replay_trades.empty: return None
            df = self.replay_trades[(self.replay_trades['exit_time'] <= current_time) & (self.replay_trades['result'] != "OPEN")].copy()
        elif self.outcomes_path.exists():
            df = pd.read_csv(self.outcomes_path, on_bad_lines="skip")
        else:
            return None

        try:
            if df.empty: return None

            # Helper to calculate stats per group
            def calc_group_stats(group_col):
                agg = df.groupby(group_col).agg(
                    trades=('pnl', 'count'),
                    wins=('pnl', lambda x: (x > 0).sum()),
                    net_pnl=('pnl', 'sum'),
                    gross_profit=('pnl', lambda x: x[x > 0].sum()),
                    gross_loss=('pnl', lambda x: abs(x[x < 0].sum()))
                )
                if agg.empty: return pd.DataFrame()
                
                agg['win_rate'] = (agg['wins'] / agg['trades']) * 100
                agg['pf'] = agg['gross_profit'] / agg['gross_loss'].replace(0, 1.0)
                agg['expectancy'] = agg['net_pnl'] / agg['trades']
                return agg.round(2).reset_index()

            # Build Matrix
            matrix = {
                "behavior": calc_group_stats("behavior_label"),
                "session": calc_group_stats("session"),
                "setup": calc_group_stats("setup"),
                "market_regime": calc_group_stats("market_regime"), # New: Group by market regime
                "alpha_range": calc_group_stats(pd.cut(df['alpha_score'], bins=[0, 75, 85, 101], labels=['Low', 'High', 'Elite'])) if 'alpha_score' in df.columns else pd.DataFrame(),
                "flow_range": calc_group_stats(pd.cut(df['flow_score'], bins=[0, 55, 75, 101], labels=['Low', 'Mid', 'High'])) if 'flow_score' in df.columns else pd.DataFrame()
            }
            
            # Weekly Governance: Top 3 and Bottom 3
            matrix["weekly_report"] = self._clean_dataframe_for_json(df.tail(50).sort_values("pnl", ascending=False))
            
            return matrix
        except Exception as e:
            logger.error("Error generating expectancy intelligence: %s", e)
            return None

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

    def get_signals(self, mode="LIVE", replay_index=0):
        """Returns latest signals for the signal feed."""
        try:
            if mode == "LIVE":
                df = get_csv_tail(self.setup_path, 50)
            elif mode == "REPLAY" and not self.replay_ohlc.empty:
                # Filter for rows that actually contain a signal intent in V3
                df = self.replay_ohlc.iloc[:replay_index + 1].copy()
                df = df[df['signal'].isin(['ALPHA', 'FLOW'])].tail(50)
            else: return pd.DataFrame()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error getting signals from state manager: {e}")
            return pd.DataFrame()
