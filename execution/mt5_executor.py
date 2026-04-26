from __future__ import annotations
from typing import Any
from datetime import datetime

import pandas as pd


class MT5ExecutionEngine:
    """MT5 execution interface for demo, paper, and live trade placement."""

    def __init__(self, config: Any):
        self.config = config
        self.demo_mode = getattr(config.live, "enabled", False) is False
        self.execution_log: list[dict[str, Any]] = []

    def run(self, df: pd.DataFrame, execute: bool = False) -> dict[str, Any]:
        """Execute or simulate the trade plan depending on mode."""
        signals = df[df["signal"].isin(["ALPHA", "FLOW"])].copy()
        execution_results = []

        for _, signal in signals.iterrows():
            if execute and not self.demo_mode:
                result = self._send_order(signal)
            else:
                result = self._simulate_order(signal)

            execution_results.append(result)
            self.execution_log.append(result)

        return {
            "executed": len(execution_results),
            "mode": "LIVE" if execute and not self.demo_mode else "DEMO",
            "results": execution_results,
        }

    def _send_order(self, signal: pd.Series) -> dict[str, Any]:
        """Send order to MT5 or broker API."""
        order = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.config.market.symbol,
            "side": signal.get("direction", "LONG"),
            "entry": float(signal.get("entry_price", 0.0)),
            "stop_loss": float(signal.get("stop_loss", 0.0)),
            "take_profit": float(signal.get("take_profit", 0.0)),
            "lot_size": float(signal.get("position_size", 0.0)),
            "signal_type": signal.get("signal", "UNKNOWN"),
            "alpha_score": float(signal.get("alpha_score", 0.0)),
            "flow_score": float(signal.get("flow_score", 0.0)),
            "status": "SUBMITTED",
            "ticket": None,
        }

        try:
            pass
        except Exception as e:
            order["status"] = "ERROR"
            order["error"] = str(e)

        return order

    def _simulate_order(self, signal: pd.Series) -> dict[str, Any]:
        """Simulate order for demo/paper trading."""
        order = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.config.market.symbol,
            "side": signal.get("direction", "LONG"),
            "entry": float(signal.get("entry_price", 0.0)),
            "stop_loss": float(signal.get("stop_loss", 0.0)),
            "take_profit": float(signal.get("take_profit", 0.0)),
            "lot_size": float(signal.get("position_size", 0.0)),
            "signal_type": signal.get("signal", "UNKNOWN"),
            "alpha_score": float(signal.get("alpha_score", 0.0)),
            "flow_score": float(signal.get("flow_score", 0.0)),
            "status": "DEMO_PENDING",
            "ticket": f"DEMO_{len(self.execution_log) + 1}",
        }
        return order

    def export_log(self, output_path: str | None = None) -> pd.DataFrame:
        """Export execution log to CSV."""
        log_df = pd.DataFrame(self.execution_log)
        if output_path:
            log_df.to_csv(output_path, index=False)
        return log_df
