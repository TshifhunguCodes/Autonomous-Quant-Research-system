from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class MarketBehaviorEngine:
    """Classifies current market behavior from price action, ATR, and momentum."""

    def __init__(self, config: Any):
        self.config = config

    def classify_market(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._build_features(out)
        out["behavior_label"] = "RANGE"
        out.loc[out["breakout"] == 1, "behavior_label"] = "BREAKOUT"
        out.loc[
            (out["reversal"] == 1) & (out["breakout"] == 0),
            "behavior_label",
        ] = "REVERSAL"
        out.loc[
            (out["volatility"] == 1) & (out["breakout"] == 0) & (out["reversal"] == 0),
            "behavior_label",
        ] = "VOLATILE"
        out.loc[
            (out["choppy"] == 1) & (out["breakout"] == 0) & (out["reversal"] == 0),
            "behavior_label",
        ] = "CHOPPY"
        out.loc[
            (out["trend_up"] == 1)
            & (out["breakout"] == 0)
            & (out["reversal"] == 0)
            & (out["volatility"] == 0),
            "behavior_label",
        ] = "TREND_UP"
        out.loc[
            (out["trend_down"] == 1)
            & (out["breakout"] == 0)
            & (out["reversal"] == 0)
            & (out["volatility"] == 0),
            "behavior_label",
        ] = "TREND_DOWN"

        out["behavior_confidence"] = out.apply(self.confidence_score, axis=1).astype(int)
        return out

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["time"] = pd.to_datetime(out["time"])
        out = out.sort_values("time").reset_index(drop=True)
        out["prev_close"] = out["close"].shift(1)
        out["momentum"] = out["close"] - out["prev_close"]
        out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["slope"] = out["ema20"].diff().fillna(0.0)
        out["high_20"] = out["high"].rolling(20, min_periods=1).max().shift(1)
        out["low_20"] = out["low"].rolling(20, min_periods=1).min().shift(1)
        tr_components = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - out["prev_close"]).abs(),
                (out["low"] - out["prev_close"]).abs(),
            ],
            axis=1,
        )
        out["tr"] = tr_components.max(axis=1)
        out["atr14"] = out["tr"].rolling(14, min_periods=1).mean()
        out["avg_tr_20"] = out["tr"].rolling(20, min_periods=1).mean()
        out["range"] = out["high"] - out["low"]
        out["range_mean"] = out["range"].rolling(20, min_periods=1).mean()
        out["candle_expansion"] = out["range"] > (out["range_mean"] * 1.25)
        out["volatility"] = (out["atr14"] > out["avg_tr_20"] * 1.25).astype(int)
        out["trend_up"] = (
            (out["close"] > out["ema20"]) & (out["slope"] > 0) & (out["close"] > out["close"].shift(5))
        ).astype(int)
        out["trend_down"] = (
            (out["close"] < out["ema20"]) & (out["slope"] < 0) & (out["close"] < out["close"].shift(5))
        ).astype(int)
        out["breakout"] = (
            (out["close"] > out["high_20"]) | (out["close"] < out["low_20"])
        ).astype(int)
        out["reversal"] = (
            (out["momentum"] * out["momentum"].shift(1) < 0)
            & out["candle_expansion"]
            & out["prev_close"].notna()
        ).astype(int)
        out["flip"] = (out["close"] > out["open"]).astype(int)
        out["flip_count_10"] = out["flip"].rolling(10, min_periods=1).apply(
            lambda values: int(np.abs(np.diff(values)).sum()), raw=True
        )
        out["choppy"] = (
            (out["range"] < out["range_mean"] * 0.8)
            & (out["flip_count_10"] >= 4)
            & (out["volatility"] == 0)
        ).astype(int)
        out["confidence_base"] = 50
        return out

    def confidence_score(self, row: pd.Series) -> int:
        score = int(row.get("confidence_base", 50))
        if row["behavior_label"] in ["TREND_UP", "TREND_DOWN"]:
            score += 18
        if row["behavior_label"] == "BREAKOUT":
            score += 18
        if row["behavior_label"] == "REVERSAL":
            score += 15
        if row["behavior_label"] == "VOLATILE":
            score += 12
        if row["behavior_label"] == "CHOPPY":
            score -= 10
        if row["candle_expansion"]:
            score += 8
        if row["volatility"]:
            score += 6
        if row["trend_up"] or row["trend_down"]:
            score += 5
        if row["reversal"]:
            score += 4
        return max(0, min(100, score))
