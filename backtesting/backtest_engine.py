from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class BacktestEngine:
    """Backtest engine for in-sample, out-of-sample, rolling, and walk-forward tests."""

    def __init__(self, config: Any):
        self.config = config

    def run(self, df: pd.DataFrame, in_sample_end: str | None = None, oos_start: str | None = None) -> dict[str, Any]:
        """Execute backtest stages and produce metrics."""
        result = {
            "summary": {},
            "trades": [],
            "equity_curve": [],
            "in_sample_metrics": {},
            "oos_metrics": {},
        }

        if in_sample_end is None or oos_start is None:
            result["full_metrics"] = self.compute_metrics(df)
            result["equity_curve"] = self._compute_equity_curve(df)
            return result

        in_sample = df[df["time"] <= pd.to_datetime(in_sample_end)]
        out_of_sample = df[df["time"] >= pd.to_datetime(oos_start)]

        result["in_sample_metrics"] = self.compute_metrics(in_sample)
        result["oos_metrics"] = self.compute_metrics(out_of_sample)
        result["equity_curve"] = self._compute_equity_curve(df)
        return result

    def compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute performance metrics like PF, win rate, drawdown, and average trade."""
        trades = self._simulate_trades(df)
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "sharpe_ratio": 0.0,
            }

        starting_equity = float(self.config.backtest.starting_balance)
        pnl_series = pd.Series([trade["pnl"] for trade in trades])

        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]

        total_wins = wins.sum()
        total_losses = abs(losses.sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0

        cumulative_equity = pd.Series(np.cumsum(pnl_series) + starting_equity)
        returns = cumulative_equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        running_max = cumulative_equity.expanding().max()
        drawdown = (cumulative_equity - running_max) / running_max
        max_dd = drawdown.min()

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "largest_win": float(wins.max()) if len(wins) > 0 else 0.0,
            "largest_loss": float(losses.min()) if len(losses) > 0 else 0.0,
            "max_drawdown": float(max_dd),
            "total_pnl": float(pnl_series.sum()),
            "sharpe_ratio": float(sharpe),
        }

    def _simulate_trades(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        df = df.sort_values("time").reset_index(drop=True)
        active_trade: dict[str, Any] | None = None
        trades: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            if active_trade is not None:
                exit_info = self._resolve_price_exit(active_trade, row)
                if exit_info is not None:
                    trades.append({**active_trade, **exit_info})
                    active_trade = None

            if active_trade is not None:
                continue

            if row.get("signal") not in ["ALPHA", "FLOW"]:
                continue

            entry_price = float(row.get("entry_price", row.get("close", 0.0)))
            stop_loss = float(row.get("stop_loss", 0.0))
            take_profit = float(row.get("take_profit", 0.0))
            position_size = float(row.get("position_size", 0.0))
            direction = str(row.get("direction", "LONG")).upper()
            side = "BUY" if direction == "LONG" else "SELL"

            if position_size <= 0 or stop_loss == 0 or take_profit == 0:
                continue

            active_trade = {
                "entry_time": row["time"],
                "signal": row.get("signal"),
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "alpha_score": float(row.get("alpha_score", 0.0)),
                "flow_score": float(row.get("flow_score", 0.0)),
            }

        return trades

    def _resolve_price_exit(self, trade: dict[str, Any], row: pd.Series) -> dict[str, Any] | None:
        high = float(row.get("high", 0.0))
        low = float(row.get("low", 0.0))
        exit_price = None
        result = None

        if trade["side"] == "BUY":
            tp_hit = high >= trade["take_profit"]
            sl_hit = low <= trade["stop_loss"]
            if tp_hit and sl_hit:
                result = "SL"
                exit_price = trade["stop_loss"]
            elif sl_hit:
                result = "SL"
                exit_price = trade["stop_loss"]
            elif tp_hit:
                result = "TP"
                exit_price = trade["take_profit"]
        else:
            tp_hit = low <= trade["take_profit"]
            sl_hit = high >= trade["stop_loss"]
            if tp_hit and sl_hit:
                result = "SL"
                exit_price = trade["stop_loss"]
            elif sl_hit:
                result = "SL"
                exit_price = trade["stop_loss"]
            elif tp_hit:
                result = "TP"
                exit_price = trade["take_profit"]

        if result is None:
            return None

        if trade["side"] == "BUY":
            pnl = (exit_price - trade["entry_price"]) * trade["position_size"]
        else:
            pnl = (trade["entry_price"] - exit_price) * trade["position_size"]

        commission = float(getattr(self.config.backtest, "commission_per_trade", 0.0))
        pnl -= commission

        return {
            "exit_time": row["time"],
            "exit_price": exit_price,
            "result": result,
            "pnl": float(round(pnl, 2)),
        }

    def _compute_equity_curve(self, df: pd.DataFrame) -> pd.Series:
        df = df.sort_values("time").reset_index(drop=True)
        active_trade: dict[str, Any] | None = None
        equity_curve: list[float] = []

        for _, row in df.iterrows():
            if active_trade is not None:
                exit_info = self._resolve_price_exit(active_trade, row)
                if exit_info is not None:
                    equity_curve.append(exit_info["pnl"])
                    active_trade = None
                    continue

            equity_curve.append(0.0)

            if row.get("signal") not in ["ALPHA", "FLOW"]:
                continue

            entry_price = float(row.get("entry_price", row.get("close", 0.0)))
            stop_loss = float(row.get("stop_loss", 0.0))
            take_profit = float(row.get("take_profit", 0.0))
            position_size = float(row.get("position_size", 0.0))
            direction = str(row.get("direction", "LONG")).upper()
            side = "BUY" if direction == "LONG" else "SELL"

            if position_size <= 0 or stop_loss == 0 or take_profit == 0:
                continue

            active_trade = {
                "entry_time": row["time"],
                "signal": row.get("signal"),
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
            }

        return pd.Series(equity_curve, index=df.index)
