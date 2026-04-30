import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
from core.logging_utils import get_logger

logger = get_logger(__name__)

class MT5Bridge:
    def __init__(self, config):
        self.config = config
        # Safe path resolution for backtest_dir with fallback to data/backtest
        backtest_dir = getattr(config.paths, "backtest_dir", Path("data/backtest"))
        
        self.audit_log_path = backtest_dir.parent / "live" / "execution_audit.csv"
        self.outcomes_log_path = backtest_dir.parent / "live" / "trade_outcomes.csv"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_and_validate(self):
        """Initializes MT5 and hard-blocks non-demo accounts."""
        # If you need automatic login, provide your credentials here:
        # login = 12345678
        # password = "your_password"
        # server = "your_broker_server"
        
        if not mt5.initialize(): # Add login, password, server here if needed
            logger.error("❌ MT5 Initialization failed")
            return False

        acc_info = mt5.account_info()
        if acc_info is None:
            logger.error("❌ Could not retrieve account info")
            return False

        # Requirement 1: Hard block live accounts
        if acc_info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            logger.critical("🚨 SECURITY BREACH: Live account detected. MT5 Bridge only allows DEMO accounts.")
            mt5.shutdown()
            return False

        logger.info("✅ MT5 Bridge connected to DEMO account: %s", acc_info.login)
        return True

    def is_candle_already_traded(self, symbol, candle_time, allow_overlap=False):
        """Checks active positions to prevent double execution on the same bar."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return False, ""

        # If overlapping positions are disabled, block if ANY position exists
        if not allow_overlap and len(positions) > 0:
            return True, "OVERLAP_BLOCKED: Overlapping positions are disabled in config."

        # Otherwise, just check if THIS specific candle has already been traded
        # This prevents opening multiple trades for the exact same 5-minute bar.
        if positions: # Re-check positions after potential early exit
            for pos in positions:
                pos_time = pd.to_datetime(pos.time, unit='s')
                current_candle_time = pd.to_datetime(candle_time)
                if pos_time >= current_candle_time:
                    return True, f"DUPLICATE_CANDLE: Position already exists for candle {candle_time}"
        return False, ""

    def check_daily_drawdown(self):
        """Checks if the daily drawdown limit has been reached."""
        max_dd = getattr(self.config.risk, "max_daily_drawdown_pct", 2.0)
        if not self.audit_log_path.exists():
            return True
            
        try:
            df = pd.read_csv(self.audit_log_path, parse_dates=["time"], on_bad_lines='skip')
            today = pd.Timestamp.now().normalize()
            # Check realized PnL from closed trades today
            daily_trades = df[(df["time"] >= today) & (df["status"].isin(["CLOSED_TP", "CLOSED_SL"]))]
            if daily_trades.empty:
                return True
                
            realized_pnl = daily_trades["pnl"].sum()
            acc_info = mt5.account_info()
            if not acc_info: return True
            
            balance = acc_info.balance
            drawdown_pct = (abs(realized_pnl) / balance) * 100 if realized_pnl < 0 else 0
            
            if drawdown_pct >= max_dd:
                logger.error("🚨 Daily Drawdown Limit Reached: %.2f%%", drawdown_pct)
                return False
            return True
        except Exception as e:
            logger.error("Error checking drawdown: %s", e)
            return True

    def check_simultaneous_positions(self):
        """Checks if the maximum number of simultaneous positions has been reached."""
        max_pos = getattr(self.config.risk, "max_simultaneous_positions", 3)
        positions = mt5.positions_total()
        if positions >= max_pos:
            logger.warning("🚫 Max positions reached: %d", positions)
            return False
        return True

    def sync_closed_trades(self):
        """Polls MT5 history to find closed trades and logs their outcomes for learning."""
        if not mt5.initialize() or not self.audit_log_path.exists():
            return

        try:
            audit_df = pd.read_csv(self.audit_log_path, parse_dates=["time"], on_bad_lines='skip')
            # Only sync trades that were EXECUTED but not yet in outcomes
            executed = audit_df[audit_df["status"] == "EXECUTED"].copy()
            
            if self.outcomes_log_path.exists():
                outcomes_df = pd.read_csv(self.outcomes_log_path, parse_dates=["entry_time"])
                # Filter out trades already processed
                executed = executed[~executed["time"].isin(outcomes_df["entry_time"])]

            if executed.empty:
                return

            # Fetch MT5 history for the last 7 days
            from_date = pd.Timestamp.now() - pd.Timedelta(days=7)
            history_deals = mt5.history_deals_get(from_date.timestamp(), pd.Timestamp.now().timestamp())
            
            if not history_deals:
                return

            deals_df = pd.DataFrame(list(history_deals), columns=history_deals[0]._as_dict().keys())
            deals_df['time'] = pd.to_datetime(deals_df['time'], unit='s')

            new_outcomes = []
            for _, trade in executed.iterrows():
                # Find matching exit deal by comment or magic
                match = deals_df[(deals_df['comment'].str.contains(str(trade['signal_time'] if 'signal_time' in trade else ""), na=False)) & 
                                 (deals_df['entry'] == 1)] # Entry 1 is 'OUT'
                
                if not match.empty:
                    outcome = match.iloc[0]
                    new_outcomes.append({
                        "entry_time": trade["time"],
                        "exit_time": outcome["time"],
                        "pnl": outcome["profit"],
                        "behavior_label": trade.get("behavior_label", "UNKNOWN"),
                        "market_regime": trade.get("market_regime", "UNKNOWN"), # Add market_regime
                        "structure_state": trade.get("structure_state", "UNKNOWN"),
                        "alpha_score": trade.get("alpha_score", 0),
                        "flow_score": trade.get("flow_score", 0),
                        "spread": trade.get("spread", 0),
                        "slippage": trade.get("slippage", 0),
                        "session": trade.get("session", "UNKNOWN"),
                        "setup": trade.get("setup", "NONE")
                    })

            if new_outcomes:
                pd.DataFrame(new_outcomes).to_csv(self.outcomes_log_path, mode='a', index=False, header=not self.outcomes_log_path.exists())
                logger.info("📈 Logged %d new trade outcomes for intelligence upgrade.", len(new_outcomes))
        except Exception as e:
            logger.error("Error syncing closed trades: %s", e)

    def execute_order(self, request, metadata):
        """Sends order to MT5 and logs result with metadata."""
        result = mt5.order_send(request)
        
        log_entry = {
            "time": pd.Timestamp.now(),
            "signal_time": metadata.get("time"),
            "symbol": request["symbol"],
            "side": "BUY" if request["type"] == mt5.ORDER_TYPE_BUY else "SELL",
            "system": metadata.get("system"),
            "regime": metadata.get("market_regime"),
            "setup": metadata.get("setup"),
            "behavior_label": metadata.get("behavior_label"),
            "structure_state": metadata.get("structure_state"),
            "zone_type": metadata.get("current_zone"),
            "alpha_score": metadata.get("alpha_score"),
            "flow_score": metadata.get("flow_score"),
            "reason_for_entry": metadata.get("execution_reason"),
            "is_exploratory": metadata.get("is_exploratory"),
            "lot": request["volume"],
            "price": request["price"],
            "status": "EXECUTED" if result.retcode == mt5.TRADE_RETCODE_DONE else "FAILED",
            "spread": metadata.get("spread", 0),
            "slippage": abs(request["price"] - metadata.get("entry_price", request["price"])),
            "retcode": result.retcode,
            "pnl": 0.0,
            "comment": request["comment"]
        }
        
        self._log_to_csv(log_entry)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_desc = {
                10016: "INVALID_STOPS (SL/TP levels are invalid, too close, or not tick-aligned)",
                10014: "INVALID_VOLUME (Check lot size or insufficient margin)",
                10030: "UNSUPPORTED_FILLING_MODE (Broker requires FOK or Return)",
                10015: "INVALID_PRICE (Market price moved too far)",
            }.get(result.retcode, f"Error Code {result.retcode}")
            logger.error("❌ Order failed! %s", error_desc)
            return False
        
        logger.info("✅ Trade Opened! Ticket: %s | System: %s", result.deal, metadata.get("system"))
        return True

    def log_blocked_trade(self, metadata, reason):
        """Logs signals that were rejected by the gate."""
        log_entry = {
            "time": pd.Timestamp.now(),
            "signal_time": metadata.get("time"),
            "symbol": self.config.market.symbol,
            "side": metadata.get("confirmed_signal", "N/A").upper(),
            "system": metadata.get("system"),
            "regime": metadata.get("market_regime"),
            "setup": metadata.get("setup"),
            "behavior_label": metadata.get("behavior_label"),
            "structure_state": metadata.get("structure_state"),
            "zone_type": metadata.get("current_zone"),
            "alpha_score": metadata.get("alpha_score"),
            "flow_score": metadata.get("flow_score"),
            "reason_for_entry": "BLOCKED_BY_GATE",
            "is_exploratory": metadata.get("is_exploratory"),
            "lot": 0,
            "price": 0,
            "status": "BLOCKED",
            "spread": metadata.get("spread", 0) / self.config.market.point_size, # Convert to points for logging
            "slippage": 0,
            "retcode": reason,
            "pnl": 0.0,
            "comment": ""
        }
        self._log_to_csv(log_entry)
        logger.warning("🚫 Trade Blocked: %s", reason)

    def _log_to_csv(self, entry):
        df = pd.DataFrame([entry])
        header = not self.audit_log_path.exists()
        df.to_csv(self.audit_log_path, mode='a', index=False, header=header)