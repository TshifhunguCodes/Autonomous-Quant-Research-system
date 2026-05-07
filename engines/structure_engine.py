from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class PriceActionStructureEngine:
    """Detects price action structure and objective pattern signatures."""

    def __init__(self, config: Any):
        self.config = config

    def build_price_action_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["time"] = pd.to_datetime(out["time"])
        out = out.sort_values("time").reset_index(drop=True)

        out["swing_high"] = (
            (out["high"] > out["high"].shift(1))
            & (out["high"] >= out["high"].shift(-1))
        )
        out["swing_low"] = (
            (out["low"] < out["low"].shift(1))
            & (out["low"] <= out["low"].shift(-1))
        )
        out["swing_high_value"] = out["high"].where(out["swing_high"], np.nan).astype(float)
        out["swing_low_value"] = out["low"].where(out["swing_low"], np.nan).astype(float)
        out["last_swing_high"] = out["swing_high_value"].ffill()
        out["last_swing_low"] = out["swing_low_value"].ffill()
        out["prev_swing_high"] = out["last_swing_high"].shift(1)
        out["prev_swing_low"] = out["last_swing_low"].shift(1)

        out["bos_up"] = (out["close"] > out["prev_swing_high"]).astype(int)
        out["bos_down"] = (out["close"] < out["prev_swing_low"]).astype(int)
        out["bos"] = ((out["bos_up"] == 1) | (out["bos_down"] == 1)).astype(int)

        out["structure_state"] = "NEUTRAL"
        out.loc[
            (out["last_swing_high"] > out["prev_swing_high"])
            & (out["last_swing_low"] > out["prev_swing_low"]),
            "structure_state",
        ] = "HH"
        out.loc[
            (out["last_swing_high"] <= out["prev_swing_high"])
            & (out["last_swing_low"] > out["prev_swing_low"]),
            "structure_state",
        ] = "HL"
        out.loc[
            (out["last_swing_high"] < out["prev_swing_high"])
            & (out["last_swing_low"] <= out["prev_swing_low"]),
            "structure_state",
        ] = "LL"
        out.loc[
            (out["last_swing_high"] < out["prev_swing_high"])
            & (out["last_swing_low"] >= out["prev_swing_low"]),
            "structure_state",
        ] = "LH"

        out["choch"] = 0
        bullish_transition = (
            (out["bos_up"] == 1)
            & out["structure_state"].shift(1).isin(["LL", "LH"])
        )
        bearish_transition = (
            (out["bos_down"] == 1)
            & out["structure_state"].shift(1).isin(["HH", "HL"])
        )
        out.loc[bullish_transition | bearish_transition, "choch"] = 1

        out["double_top"] = (
            out["swing_high"]
            & out["prev_swing_high"].notna()
            & (out["high"].sub(out["prev_swing_high"]).abs() <= out["range"].rolling(5, min_periods=1).mean() * 0.2)
            & (out["close"] < out["prev_close"])
        ).astype(int)
        out["double_bottom"] = (
            out["swing_low"]
            & out["prev_swing_low"].notna()
            & (out["low"].sub(out["prev_swing_low"]).abs() <= out["range"].rolling(5, min_periods=1).mean() * 0.2)
            & (out["close"] > out["prev_close"])
        ).astype(int)

        out["break_retest"] = 0
        out.loc[
            (out["bos_up"] == 1)
            & (out["prev_swing_high"].notna())
            & (out["low"] <= out["prev_swing_high"] * 1.005)
            & (out["low"] >= out["prev_swing_high"] * 0.995),
            "break_retest",
        ] = 1
        out.loc[
            (out["bos_down"] == 1)
            & (out["prev_swing_low"].notna())
            & (out["high"] >= out["prev_swing_low"] * 0.995)
            & (out["high"] <= out["prev_swing_low"] * 1.005),
            "break_retest",
        ] = 1

        out["pattern"] = "NONE"
        out.loc[out["double_top"] == 1, "pattern"] = "DOUBLE_TOP"
        out.loc[out["double_bottom"] == 1, "pattern"] = "DOUBLE_BOTTOM"
        out.loc[out["break_retest"] == 1, "pattern"] = "BREAK_RETEST"
        out.loc[(out["choch"] == 1) & (out["pattern"] == "NONE"), "pattern"] = "CHOCH"
        out = self._classify_retracement(out)
        return out

    def identify_pattern(self, row: pd.Series) -> str:
        if row.get("double_top", 0) == 1:
            return "DOUBLE_TOP"
        if row.get("double_bottom", 0) == 1:
            return "DOUBLE_BOTTOM"
        if row.get("break_retest", 0) == 1:
            return "BREAK_RETEST"
        if row.get("choch", 0) == 1:
            return "CHOCH"
        return "NONE"

    def _classify_retracement(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        trend_up_mask = out["behavior_label"] == "TREND_UP"
        trend_down_mask = out["behavior_label"] == "TREND_DOWN"

        out["higher_low_holding"] = ((out["last_swing_low"] >= out["prev_swing_low"]) | out["prev_swing_low"].isna()).astype(int)
        out["lower_high_holding"] = ((out["last_swing_high"] <= out["prev_swing_high"]) | out["prev_swing_high"].isna()).astype(int)

        out["fib_anchor_start"] = np.where(trend_up_mask, out["prev_swing_low"], out["prev_swing_high"])
        out["fib_anchor_end"] = np.where(trend_up_mask, out["prev_swing_high"], out["prev_swing_low"])
        out["fib_range"] = (out["fib_anchor_end"] - out["fib_anchor_start"]).abs()

        retracement_price = np.where(trend_up_mask, out["close"], np.where(trend_down_mask, out["close"], np.nan))
        retracement_distance = np.where(
            trend_up_mask,
            out["fib_anchor_end"] - retracement_price,
            np.where(trend_down_mask, retracement_price - out["fib_anchor_end"], np.nan),
        )
        out["fib_retracement_pct"] = np.where(
            out["fib_range"] > 0,
            (retracement_distance / out["fib_range"]).clip(lower=0.0) * 100.0,
            np.nan,
        )

        out["reversal_warning"] = 0
        out.loc[
            (trend_up_mask | trend_down_mask)
            & out["fib_retracement_pct"].gt(78.6)
            & (out["bos"] == 0),
            "reversal_warning",
        ] = 1

        out["confirmed_reversal"] = 0
        out.loc[
            trend_up_mask
            & out["fib_retracement_pct"].gt(78.6)
            & (out["bos_down"] == 1)
            & (out["choch"] == 1)
            & (out["close"] < out["last_swing_low"]),
            "confirmed_reversal",
        ] = 1
        out.loc[
            trend_down_mask
            & out["fib_retracement_pct"].gt(78.6)
            & (out["bos_up"] == 1)
            & (out["choch"] == 1)
            & (out["close"] > out["last_swing_high"]),
            "confirmed_reversal",
        ] = 1

        out["retracement_class"] = "NON_TREND"
        out.loc[(trend_up_mask | trend_down_mask) & out["fib_retracement_pct"].between(0.0, 38.0, inclusive="both"), "retracement_class"] = "SHALLOW_CONTINUATION"
        out.loc[(trend_up_mask | trend_down_mask) & out["fib_retracement_pct"].gt(38.0) & out["fib_retracement_pct"].le(61.8), "retracement_class"] = "NORMAL_CONTINUATION"
        out.loc[(trend_up_mask | trend_down_mask) & out["fib_retracement_pct"].gt(61.8) & out["fib_retracement_pct"].le(78.6), "retracement_class"] = "DEEP_CONTINUATION"
        out.loc[(trend_up_mask | trend_down_mask) & out["reversal_warning"].eq(1), "retracement_class"] = "REVERSAL_WARNING"
        out.loc[out["confirmed_reversal"] == 1, "retracement_class"] = "CONFIRMED_REVERSAL"

        out["retracement_trade_allowed"] = 1
        out.loc[out["retracement_class"] == "REVERSAL_WARNING", "retracement_trade_allowed"] = 0
        out.loc[
            (trend_up_mask & (out["retracement_class"] == "DEEP_CONTINUATION") & (out["higher_low_holding"] == 0))
            | (trend_down_mask & (out["retracement_class"] == "DEEP_CONTINUATION") & (out["lower_high_holding"] == 0)),
            "retracement_trade_allowed",
        ] = 0
        return out
