import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
import csv
from core.logging_utils import get_logger
from export_mt5_system_history import build_position_history, rows_from_mt5_tuples

logger = get_logger(__name__)

AUDIT_COLUMNS = [
    "time",
    "signal_time",
    "symbol",
    "side",
    "system",
    "regime",
    "setup",
    "behavior_label",
    "structure_state",
    "zone_type",
    "alpha_score",
    "flow_score",
    "reason_for_entry",
    "is_exploratory",
    "lot",
    "price",
    "status",
    "spread",
    "slippage",
    "retcode",
    "order_ticket",
    "deal_ticket",
    "pnl",
    "comment",
    "visual_zone_score",
    "visual_zone_direction",
    "visual_zone_type",
    "visual_zone_reason",
]

class MT5Bridge:
    def __init__(self, config):
        self.config = config
        # Safe path resolution for backtest_dir with fallback to data/backtest
        backtest_dir = getattr(config.paths, "backtest_dir", Path("data/backtest"))
        
        self.audit_log_path = backtest_dir.parent / "live" / "execution_audit.csv"
        self.outcomes_log_path = backtest_dir.parent / "live" / "trade_outcomes.csv"
        self.mt5_history_path = backtest_dir.parent / "live" / "mt5_system_trade_history.csv"
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

    def _is_system_position(self, position) -> bool:
        """Return True only for AQRS positions on this system's configured market."""
        configured_symbol = str(getattr(self.config.market, "symbol", "") or "").upper()
        position_symbol = str(getattr(position, "symbol", "") or "").upper()
        comment = str(getattr(position, "comment", "") or "").upper()
        magic = getattr(position, "magic", None)
        system_magic = int(getattr(self.config.market, "magic_number", 202404))

        symbol_match = (
            configured_symbol
            and (
                configured_symbol == position_symbol
                or configured_symbol in position_symbol
                or (configured_symbol in {"XAUUSD", "GOLD"} and ("XAU" in position_symbol or "GOLD" in position_symbol))
            )
        )
        try:
            magic_match = magic is not None and int(magic) == system_magic
        except (TypeError, ValueError):
            magic_match = False
        system_tag_match = comment.startswith("AQ_") or magic_match
        return bool(symbol_match and system_tag_match)

    def get_system_positions(self, symbol: str | None = None):
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return []
        return [position for position in positions if self._is_system_position(position)]

    def is_candle_already_traded(self, symbol, candle_time, allow_overlap=False):
        """Checks active positions to prevent double execution on the same bar."""
        positions = self.get_system_positions(symbol=symbol)
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
        if not self.outcomes_log_path.exists():
            return True
            
        try:
            df = pd.read_csv(self.outcomes_log_path, parse_dates=["exit_time"], on_bad_lines='skip')
            today = pd.Timestamp.now().normalize()
            # Check realized PnL from closed trades today
            daily_trades = df[df["exit_time"] >= today]
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
        positions = self.get_system_positions()
        if len(positions) >= max_pos:
            logger.warning("AQRS max positions reached: %d", len(positions))
            return False
        return True

    def _legacy_sync_closed_trades(self):
        """Old matcher retained for reference; sync_closed_trades below is used."""
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

    def sync_closed_trades(self):
        """Poll MT5 history and write closed trade PnL to trade_outcomes.csv."""
        if not mt5.initialize() or not self.audit_log_path.exists():
            return

        try:
            self.export_system_trade_history(days=90)
            audit_df = self._read_audit_log()
            if audit_df.empty:
                return

            executed = audit_df[audit_df["status"] == "EXECUTED"].copy()
            executed["time"] = pd.to_datetime(executed["time"], errors="coerce")
            executed["signal_time"] = pd.to_datetime(executed["signal_time"], errors="coerce")
            executed = executed.dropna(subset=["time"])

            if self.outcomes_log_path.exists():
                outcomes_df = pd.read_csv(self.outcomes_log_path, parse_dates=["entry_time"], on_bad_lines="skip")
                executed = executed[~executed["time"].isin(outcomes_df["entry_time"])]
                if "entry_deal" in outcomes_df.columns and "deal_ticket" in executed.columns:
                    used_deals = set(pd.to_numeric(outcomes_df["entry_deal"], errors="coerce").dropna().astype("int64"))
                    deal_tickets = pd.to_numeric(executed["deal_ticket"], errors="coerce")
                    executed = executed[~deal_tickets.isin(used_deals)]

            if executed.empty:
                return

            oldest_entry = executed["time"].min() - pd.Timedelta(hours=1)
            from_date = min(oldest_entry, pd.Timestamp.now() - pd.Timedelta(days=14))
            history_deals = mt5.history_deals_get(
                from_date.to_pydatetime(),
                pd.Timestamp.now().to_pydatetime(),
            )
            if not history_deals:
                return

            deals_df = pd.DataFrame([deal._asdict() for deal in history_deals])
            if deals_df.empty:
                return
            deals_df["time"] = pd.to_datetime(deals_df["time"], unit="s")

            new_outcomes = []
            used_entry_deals = set()
            if self.outcomes_log_path.exists():
                try:
                    prior = pd.read_csv(self.outcomes_log_path, usecols=["entry_deal"], on_bad_lines="skip")
                    used_entry_deals.update(pd.to_numeric(prior["entry_deal"], errors="coerce").dropna().astype("int64").tolist())
                except Exception:
                    pass

            for _, trade in executed.iterrows():
                outcome = self._match_closed_trade(trade, deals_df)
                if outcome is None:
                    continue
                try:
                    entry_deal_key = int(outcome["entry_deal"])
                except (TypeError, ValueError):
                    entry_deal_key = None
                if entry_deal_key is not None and entry_deal_key in used_entry_deals:
                    continue
                if entry_deal_key is not None:
                    used_entry_deals.add(entry_deal_key)

                new_outcomes.append({
                    "entry_time": trade["time"],
                    "signal_time": trade.get("signal_time"),
                    "exit_time": outcome["exit_time"],
                    "symbol": trade.get("symbol", self.config.market.symbol),
                    "side": trade.get("side", "UNKNOWN"),
                    "system": trade.get("system", "UNKNOWN"),
                    "result": "WIN" if outcome["pnl"] > 0 else ("LOSS" if outcome["pnl"] < 0 else "BREAKEVEN"),
                    "pnl": outcome["pnl"],
                    "entry_price": trade.get("price", 0),
                    "exit_price": outcome["exit_price"],
                    "position_id": outcome["position_id"],
                    "entry_deal": outcome["entry_deal"],
                    "exit_deals": outcome["exit_deals"],
                    "behavior_label": trade.get("behavior_label", "UNKNOWN"),
                    "market_regime": trade.get("regime", trade.get("market_regime", "UNKNOWN")),
                    "structure_state": trade.get("structure_state", "UNKNOWN"),
                    "alpha_score": trade.get("alpha_score", 0),
                    "flow_score": trade.get("flow_score", 0),
                    "spread": trade.get("spread", 0),
                    "slippage": trade.get("slippage", 0),
                    "session": trade.get("session", "UNKNOWN"),
                    "setup": trade.get("setup", "NONE"),
                    "comment": trade.get("comment", ""),
                    "visual_zone_score": trade.get("visual_zone_score", 0),
                    "visual_zone_direction": trade.get("visual_zone_direction", "NEUTRAL"),
                    "visual_zone_type": trade.get("visual_zone_type", "NONE"),
                    "visual_zone_reason": trade.get("visual_zone_reason", ""),
                })

            if new_outcomes:
                pd.DataFrame(new_outcomes).to_csv(
                    self.outcomes_log_path,
                    mode="a",
                    index=False,
                    header=not self.outcomes_log_path.exists(),
                )
                try:
                    from strategy.execution_gate import ExecutionGate

                    for outcome in new_outcomes:
                        ExecutionGate.record_trade_outcome(outcome, float(outcome.get("pnl", 0.0) or 0.0))
                except Exception as e:
                    logger.warning("Adaptive outcome update failed: %s", e)
                logger.info("Logged %d closed trade outcomes.", len(new_outcomes))
        except Exception as e:
            logger.error("Error syncing closed trades: %s", e)

    def export_system_trade_history(self, days=90):
        """Refresh broker-truth AQRS trade history from MT5 into a clean CSV."""
        try:
            from_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            to_date = pd.Timestamp.now()
            deals = mt5.history_deals_get(from_date.to_pydatetime(), to_date.to_pydatetime())
            deals_df = rows_from_mt5_tuples(deals)
            if deals_df.empty:
                return pd.DataFrame()

            symbol = str(getattr(self.config.market, "symbol", ""))
            if symbol and "symbol" in deals_df.columns:
                deals_df = deals_df[deals_df["symbol"].astype(str).eq(symbol)]

            magic = int(getattr(self.config.market, "magic_number", 202404))
            history = build_position_history(deals_df, magic=magic, comment_prefix="AQ_")
            self.mt5_history_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(self.mt5_history_path, index=False)
            logger.info("MT5 system trade history refreshed: %d rows", len(history))
            return history
        except Exception as e:
            logger.warning("MT5 system trade history export failed: %s", e)
            return pd.DataFrame()

    def _read_audit_log(self):
        """Read old and new execution_audit rows without dropping mixed-schema entries."""
        if not self.audit_log_path.exists():
            return pd.DataFrame(columns=AUDIT_COLUMNS)

        rows = []
        with self.audit_log_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            for raw in reader:
                if len(raw) >= len(AUDIT_COLUMNS):
                    rows.append(raw[:len(AUDIT_COLUMNS)])
                    continue

                row_map = dict(zip(header, raw))
                rows.append([row_map.get(col, "") for col in AUDIT_COLUMNS])

        df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
        for col in ["time", "signal_time"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in ["alpha_score", "flow_score", "lot", "price", "spread", "slippage", "pnl", "visual_zone_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _match_closed_trade(self, trade, deals_df):
        """Match one audit entry to a closed MT5 position and summarize realized PnL."""
        symbol = str(trade.get("symbol", self.config.market.symbol))
        comment = str(trade.get("comment", "") or "")
        entry_time = pd.to_datetime(trade.get("time"), errors="coerce")
        if pd.isna(entry_time):
            return None

        candidates = deals_df[deals_df["time"] >= entry_time - pd.Timedelta(minutes=10)].copy()
        if "symbol" in candidates.columns:
            candidates = candidates[candidates["symbol"].astype(str) == symbol]

        entry_code = getattr(mt5, "DEAL_ENTRY_IN", 0)
        exit_code = getattr(mt5, "DEAL_ENTRY_OUT", 1)
        entry_col = candidates["entry"] if "entry" in candidates.columns else pd.Series(dtype=int)
        entry_deals = candidates[entry_col == entry_code].copy()
        exact_deal = pd.to_numeric(pd.Series([trade.get("deal_ticket", None)]), errors="coerce").iloc[0]
        if pd.notna(exact_deal):
            exact_match = entry_deals[pd.to_numeric(entry_deals.get("ticket"), errors="coerce") == int(exact_deal)]
            if not exact_match.empty:
                entry_deals = exact_match
        elif "comment" in candidates.columns and comment:
            comment_matches = entry_deals[entry_deals["comment"].astype(str) == comment]
            if not comment_matches.empty:
                entry_deals = comment_matches
        if entry_deals.empty:
            return None

        entry_deals["time_diff"] = (entry_deals["time"] - entry_time).abs()
        entry_deal = entry_deals.sort_values("time_diff").iloc[0]
        position_id = entry_deal.get("position_id", entry_deal.get("position", None))
        if pd.isna(position_id):
            return None

        pos_col = deals_df["position_id"] if "position_id" in deals_df.columns else deals_df.get("position")
        if pos_col is None or "entry" not in deals_df.columns:
            return None
        all_entries = deals_df["entry"]
        exits = deals_df[
            (pos_col == position_id)
            & (all_entries == exit_code)
            & (deals_df["time"] >= entry_deal["time"])
        ].copy()
        if exits.empty:
            return None

        pnl_cols = [col for col in ["profit", "commission", "swap", "fee"] if col in exits.columns]
        pnl = float(exits[pnl_cols].fillna(0).sum().sum()) if pnl_cols else 0.0
        exit_price = float(exits.sort_values("time").iloc[-1].get("price", 0.0) or 0.0)
        exit_deals = ",".join(str(int(ticket)) for ticket in exits.get("ticket", pd.Series(dtype=int)).dropna())

        return {
            "pnl": pnl,
            "exit_time": exits["time"].max(),
            "exit_price": exit_price,
            "position_id": position_id,
            "entry_deal": entry_deal.get("ticket", ""),
            "exit_deals": exit_deals,
        }

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
            "order_ticket": getattr(result, "order", 0),
            "deal_ticket": getattr(result, "deal", 0),
            "pnl": 0.0,
            "comment": request["comment"],
            "visual_zone_score": metadata.get("visual_zone_score", 0),
            "visual_zone_direction": metadata.get("visual_zone_direction", "NEUTRAL"),
            "visual_zone_type": metadata.get("visual_zone_type", "NONE"),
            "visual_zone_reason": metadata.get("visual_zone_reason", ""),
        }
        
        self._log_to_csv(log_entry)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_desc = {
                10016: "INVALID_STOPS (SL/TP levels are invalid, too close, or not tick-aligned)",
                10014: "INVALID_VOLUME (Check lot size or insufficient margin)",
                10030: "UNSUPPORTED_FILLING_MODE (Broker requires FOK or Return)",
                10015: "INVALID_PRICE (Market price moved too far)",
                10027: "CLIENT_DISABLES_AT (MT5 automatic trading is disabled). Enable Algo Trading/AutoTrading in terminal.",
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
            "order_ticket": 0,
            "deal_ticket": 0,
            "pnl": 0.0,
            "comment": "",
            "visual_zone_score": metadata.get("visual_zone_score", 0),
            "visual_zone_direction": metadata.get("visual_zone_direction", "NEUTRAL"),
            "visual_zone_type": metadata.get("visual_zone_type", "NONE"),
            "visual_zone_reason": metadata.get("visual_zone_reason", ""),
        }
        self._log_to_csv(log_entry)
        logger.warning("🚫 Trade Blocked: %s", reason)

    def _log_to_csv(self, entry):
        df = pd.DataFrame([entry])
        header = not self.audit_log_path.exists()
        df.to_csv(self.audit_log_path, mode='a', index=False, header=header)
