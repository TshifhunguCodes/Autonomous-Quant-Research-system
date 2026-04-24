import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
from core.logging_utils import get_logger

logger = get_logger(__name__)

class MT5Bridge:
    def __init__(self, config):
        self.config = config
        self.audit_log_path = config.paths.backtest_dir.parent / "live" / "execution_audit.csv"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_and_validate(self):
        """Initializes MT5 and hard-blocks non-demo accounts."""
        if not mt5.initialize():
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

    def is_candle_already_traded(self, symbol, candle_time):
        """Checks active positions to prevent double execution on the same bar."""
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                pos_time = pd.to_datetime(pos.time, unit='s')
                current_candle_time = pd.to_datetime(candle_time)
                if pos_time >= current_candle_time:
                    return True
        return False

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
            "is_exploratory": metadata.get("is_exploratory"),
            "lot": request["volume"],
            "price": request["price"],
            "status": "EXECUTED" if result.retcode == mt5.TRADE_RETCODE_DONE else "FAILED",
            "retcode": result.retcode,
            "comment": request["comment"]
        }
        
        self._log_to_csv(log_entry)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("❌ Order failed! Code: %s", result.retcode)
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
            "is_exploratory": metadata.get("is_exploratory"),
            "lot": 0,
            "price": 0,
            "status": "BLOCKED",
            "retcode": reason,
            "comment": ""
        }
        self._log_to_csv(log_entry)
        logger.warning("🚫 Trade Blocked: %s", reason)

    def _log_to_csv(self, entry):
        df = pd.DataFrame([entry])
        header = not self.audit_log_path.exists()
        df.to_csv(self.audit_log_path, mode='a', index=False, header=header)