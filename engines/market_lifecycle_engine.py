from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class MarketLifecycleEngine:
    """Classifies trend maturity and exhaustion using lightweight price-action features."""

    def __init__(self, config: Any):
        self.config = config

    def classify_lifecycle(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["range_mean_5"] = out["range"].rolling(5, min_periods=1).mean()
        out["range_mean_20"] = out["range"].rolling(20, min_periods=1).mean()
        out["atr_slope"] = out["atr14"].diff().fillna(0.0)
        out["atr_slowdown"] = (out["atr_slope"] < 0).astype(int)
        out["expansion_decay"] = (out["range_mean_5"] < out["range_mean_20"]).astype(int)

        candle_body = (out["close"] - out["open"]).abs()
        upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
        lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
        out["wick_rejection"] = (
            (upper_wick > candle_body * 1.25) | (lower_wick > candle_body * 1.25)
        ).astype(int)

        trend_up = out["behavior_label"] == "TREND_UP"
        trend_down = out["behavior_label"] == "TREND_DOWN"
        out["momentum_delta_3"] = out["momentum"].rolling(3, min_periods=1).mean()
        out["momentum_weakening"] = (
            (trend_up & (out["momentum_delta_3"] <= 0))
            | (trend_down & (out["momentum_delta_3"] >= 0))
        ).astype(int)

        out["failed_continuation"] = (
            ((trend_up | trend_down) & (out["retracement_class"] == "DEEP_CONTINUATION") & (out["retracement_trade_allowed"] == 0))
            | ((trend_up | trend_down) & (out["break_retest"] == 1) & (out["bos"] == 0))
        ).astype(int)

        out["failed_bos"] = (
            ((trend_up & (out["bos_up"] == 1) & (out["close"] <= out["prev_swing_high"].fillna(out["close"] - 1))))
            | ((trend_down & (out["bos_down"] == 1) & (out["close"] >= out["prev_swing_low"].fillna(out["close"] + 1))))
        ).astype(int)

        out["volatility_compression"] = (
            (out["atr14"] <= out["atr14"].rolling(20, min_periods=1).mean() * 0.85)
            & (out["range_mean_5"] <= out["range_mean_20"] * 0.85)
        ).astype(int)

        out["breakout_expansion_signal"] = (
            (out["bos"] == 1)
            & (out["range"] >= out["range_mean_20"] * 1.2)
            & (out["atr14"] >= out["atr14"].rolling(20, min_periods=1).mean())
        ).astype(int)

        out["continuation_strength"] = 50.0
        out.loc[out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]), "continuation_strength"] += 15
        out.loc[out["retracement_class"].isin(["SHALLOW_CONTINUATION", "NORMAL_CONTINUATION"]), "continuation_strength"] += 12
        out.loc[out["breakout_expansion_signal"] == 1, "continuation_strength"] += 15
        out.loc[out["failed_continuation"] == 1, "continuation_strength"] -= 20
        out.loc[out["failed_bos"] == 1, "continuation_strength"] -= 15
        out.loc[out["momentum_weakening"] == 1, "continuation_strength"] -= 10
        out["continuation_strength"] = out["continuation_strength"].clip(lower=0.0, upper=100.0)

        out["exhaustion_score"] = 0.0
        out.loc[out["expansion_decay"] == 1, "exhaustion_score"] += 20
        out.loc[out["atr_slowdown"] == 1, "exhaustion_score"] += 15
        out.loc[out["wick_rejection"] == 1, "exhaustion_score"] += 15
        out.loc[out["failed_continuation"] == 1, "exhaustion_score"] += 20
        out.loc[out["failed_bos"] == 1, "exhaustion_score"] += 15
        out.loc[out["momentum_weakening"] == 1, "exhaustion_score"] += 15
        out.loc[out["volatility_compression"] == 1, "exhaustion_score"] += 10
        out.loc[out["retracement_class"] == "REVERSAL_WARNING", "exhaustion_score"] += 20
        out.loc[out["confirmed_reversal"] == 1, "exhaustion_score"] += 25
        out["exhaustion_score"] = out["exhaustion_score"].clip(lower=0.0, upper=100.0)

        out["trend_health_score"] = (100.0 - out["exhaustion_score"] * 0.6 + out["continuation_strength"] * 0.4).clip(lower=0.0, upper=100.0)

        out["lifecycle_state"] = "TREND_HEALTHY"
        out.loc[out["volatility_compression"] == 1, "lifecycle_state"] = "RANGE_COMPRESSION"
        out.loc[out["breakout_expansion_signal"] == 1, "lifecycle_state"] = "BREAKOUT_EXPANSION"
        out.loc[
            out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]) & out["retracement_class"].eq("SHALLOW_CONTINUATION") & out["continuation_strength"].ge(70),
            "lifecycle_state",
        ] = "TREND_START"
        out.loc[
            out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]) & out["continuation_strength"].between(55, 75, inclusive="left"),
            "lifecycle_state",
        ] = "TREND_HEALTHY"
        out.loc[
            out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]) & out["continuation_strength"].between(40, 55, inclusive="left"),
            "lifecycle_state",
        ] = "TREND_EXTENDED"
        out.loc[
            out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]) & out["exhaustion_score"].between(55, 75, inclusive="left"),
            "lifecycle_state",
        ] = "TREND_EXHAUSTING"
        out.loc[
            out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]) & ((out["retracement_class"] == "REVERSAL_WARNING") | out["exhaustion_score"].ge(75)),
            "lifecycle_state",
        ] = "REVERSAL_WATCH"
        out.loc[out["confirmed_reversal"] == 1, "lifecycle_state"] = "REVERSAL_CONFIRMED"
        return out
