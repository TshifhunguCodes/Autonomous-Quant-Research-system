from __future__ import annotations
from typing import Any

import pandas as pd

from engines.behavior_engine import MarketBehaviorEngine
from engines.structure_engine import PriceActionStructureEngine
from engines.zone_engine import ZoneEngine
from risk.risk_manager import RiskManager
from systems.alpha_system import AlphaSystem
from systems.flow_system import FlowSystem


class ReplayEngine:
    """Replay engine that steps through historical bars and updates the V3 pipeline."""

    def __init__(self, config: Any):
        self.config = config
        self.behavior = MarketBehaviorEngine(config)
        self.structure = PriceActionStructureEngine(config)
        self.zone = ZoneEngine(config)
        self.alpha = AlphaSystem(config)
        self.flow = FlowSystem(config)
        self.risk = RiskManager(config)
        self.history: pd.DataFrame = pd.DataFrame()
        self.positions: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.equity: float = float(getattr(config.backtest, "starting_balance", 1000.0))
        self.equity_curve: list[float] = []

    def run(
        self,
        df: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        max_candles: int | None = None,
    ) -> pd.DataFrame:
        source = df.copy()
        source["time"] = pd.to_datetime(source["time"])
        source = source.sort_values("time").reset_index(drop=True)

        if start is not None:
            source = source[source["time"] >= pd.to_datetime(start)]
        if end is not None:
            source = source[source["time"] <= pd.to_datetime(end)]
        if max_candles is not None:
            source = source.head(max_candles)

        self.history = pd.DataFrame()
        self.positions = []
        self.trades = []
        self.equity = float(getattr(self.config.backtest, "starting_balance", 1000.0))
        self.equity_curve = []

        replay_rows: list[dict[str, Any]] = []
        for _, candle in source.iterrows():
            summary = self.step(candle)
            replay_rows.append(summary)
            self.equity_curve.append(self.equity)

        return pd.DataFrame(replay_rows)

    def step(self, candle: pd.Series) -> dict[str, Any]:
        new_row = candle.to_frame().T.reset_index(drop=True)
        self.history = pd.concat([self.history, new_row], ignore_index=True)

        pipeline = self.behavior.classify_market(self.history)
        pipeline = self.structure.build_price_action_structure(pipeline)
        pipeline = self.zone.build_zones(pipeline)
        pipeline = self.alpha.generate_alpha_setups(pipeline)
        pipeline = self.flow.generate_flow_setups(pipeline)
        pipeline = self._resolve_signal(pipeline)
        pipeline = self.risk.annotate_trade_risk(pipeline)

        latest = pipeline.iloc[-1].to_dict()
        trade_event = self._update_positions(latest)
        if latest.get("signal") in ["ALPHA", "FLOW"]:
            self._open_position(latest)

        record = {
            "time": latest.get("time"),
            "behavior": latest.get("behavior_label"),
            "structure": latest.get("structure_state"),
            "pattern": latest.get("pattern"),
            "session": latest.get("session"),
            "alpha_score": latest.get("alpha_score"),
            "flow_score": latest.get("flow_score"),
            "signal": latest.get("signal"),
            "equity": self.equity,
            "open_positions": len(self.positions),
            "last_trade_event": trade_event,
        }
        return record

    def _resolve_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = "NO_TRADE"
        out.loc[out["alpha_signal"] == "ALPHA", "signal"] = "ALPHA"
        flow_mask = (out["signal"] == "NO_TRADE") & (out["flow_signal"] == "FLOW")
        out.loc[flow_mask, "signal"] = "FLOW"
        return out

    def _open_position(self, latest: dict[str, Any]) -> None:
        if latest.get("signal") == "NO_TRADE":
            return
        side = "BUY" if latest.get("behavior_label") == "TREND_UP" else "SELL"
        if latest.get("behavior_label") == "TREND_DOWN":
            side = "SELL"
        position = {
            "opened_at": latest.get("time"),
            "side": side,
            "entry": float(latest.get("close", 0.0)),
            "stop_loss": float(latest.get("stop_loss", 0.0)),
            "take_profit": float(latest.get("take_profit", 0.0)),
            "risk_amount": float(latest.get("position_risk", 0.0)),
            "signal": latest.get("signal"),
            "alpha_score": float(latest.get("alpha_score", 0.0)),
            "flow_score": float(latest.get("flow_score", 0.0)),
        }
        self.positions.append(position)

    def _update_positions(self, latest: dict[str, Any]) -> dict[str, Any]:
        if not self.positions:
            return {}

        current_low = float(latest.get("low", 0.0))
        current_high = float(latest.get("high", 0.0))
        current_time = latest.get("time")
        event: dict[str, Any] = {}
        remaining_positions: list[dict[str, Any]] = []

        for position in self.positions:
            closed = False
            if position["side"] == "BUY":
                if current_low <= position["stop_loss"]:
                    pnl = (position["stop_loss"] - position["entry"]) * position["risk_amount"]
                    self.equity += pnl
                    event = {"closed": "SL", "pnl": float(pnl), "time": current_time}
                    closed = True
                elif current_high >= position["take_profit"]:
                    pnl = (position["take_profit"] - position["entry"]) * position["risk_amount"]
                    self.equity += pnl
                    event = {"closed": "TP", "pnl": float(pnl), "time": current_time}
                    closed = True
            else:
                if current_high >= position["stop_loss"]:
                    pnl = (position["entry"] - position["stop_loss"]) * position["risk_amount"]
                    self.equity += pnl
                    event = {"closed": "SL", "pnl": float(pnl), "time": current_time}
                    closed = True
                elif current_low <= position["take_profit"]:
                    pnl = (position["entry"] - position["take_profit"]) * position["risk_amount"]
                    self.equity += pnl
                    event = {"closed": "TP", "pnl": float(pnl), "time": current_time}
                    closed = True
            if not closed:
                remaining_positions.append(position)
            else:
                self.trades.append({**position, **event})

        self.positions = remaining_positions
        return event
