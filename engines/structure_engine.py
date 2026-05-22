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
        out["time"] = pd.to_datetime(out["time"]).astype("datetime64[s]")
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
        out = self._classify_market_story(out)
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

    def _classify_market_story(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        index = out.index

        atr = out.get("atr", out["range"].rolling(14, min_periods=1).mean()).fillna(out["range"].rolling(14, min_periods=1).mean())
        avg_range = out["range"].rolling(20, min_periods=1).mean()
        tolerance = (atr * 0.25).fillna(avg_range * 0.5).clip(lower=avg_range * 0.2)

        out["broken_structure_level"] = np.nan
        out.loc[out["bos_up"].eq(1), "broken_structure_level"] = out["prev_swing_high"]
        out.loc[out["bos_down"].eq(1), "broken_structure_level"] = out["prev_swing_low"]
        out["last_broken_high"] = out["prev_swing_high"].where(out["bos_up"].eq(1), np.nan).ffill()
        out["last_broken_low"] = out["prev_swing_low"].where(out["bos_down"].eq(1), np.nan).ffill()
        out["last_break_direction"] = np.select(
            [out["bos_up"].eq(1), out["bos_down"].eq(1)],
            ["UP", "DOWN"],
            default="NONE",
        )
        out["last_break_direction"] = pd.Series(out["last_break_direction"], index=index).replace("NONE", np.nan).ffill().fillna("NONE")

        bos_seen = out["bos"].eq(1)
        out["bars_since_bos"] = bos_seen.groupby(bos_seen.cumsum()).cumcount()
        out.loc[~bos_seen.cumsum().astype(bool), "bars_since_bos"] = 999

        body = (out["close"] - out["open"]).abs()
        upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
        lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
        bullish_candle = out["close"] > out["open"]
        bearish_candle = out["close"] < out["open"]
        bullish_rejection = bullish_candle & (lower_wick >= body * 0.8)
        bearish_rejection = bearish_candle & (upper_wick >= body * 0.8)

        out["retest_broken_high"] = (
            out["last_broken_high"].notna()
            & out["low"].le(out["last_broken_high"] + tolerance)
            & out["close"].ge(out["last_broken_high"] - tolerance)
            & out["bars_since_bos"].between(1, 30, inclusive="both")
        ).astype(int)
        out["retest_broken_low"] = (
            out["last_broken_low"].notna()
            & out["high"].ge(out["last_broken_low"] - tolerance)
            & out["close"].le(out["last_broken_low"] + tolerance)
            & out["bars_since_bos"].between(1, 30, inclusive="both")
        ).astype(int)

        out["bullish_zone_rejection"] = (
            bullish_rejection
            & (
                out["retest_broken_high"].eq(1)
                | out.get("demand_zone", pd.Series(0, index=index)).eq(1)
                | out.get("is_support", pd.Series(0, index=index)).eq(1)
                | out.get("order_block", pd.Series(0, index=index)).eq(1)
                | out.get("fvg_zone", pd.Series(0, index=index)).eq(1)
            )
        ).astype(int)
        out["bearish_zone_rejection"] = (
            bearish_rejection
            & (
                out["retest_broken_low"].eq(1)
                | out.get("supply_zone", pd.Series(0, index=index)).eq(1)
                | out.get("is_resistance", pd.Series(0, index=index)).eq(1)
                | out.get("order_block", pd.Series(0, index=index)).eq(1)
                | out.get("fvg_zone", pd.Series(0, index=index)).eq(1)
            )
        ).astype(int)

        displacement = (body >= atr * 1.2) | (out.get("tick_volume", pd.Series(0, index=index)) >= out.get("tick_volume", pd.Series(0, index=index)).rolling(20, min_periods=1).mean() * 1.25)
        out["momentum_breakout"] = ((out["bos"].eq(1)) & displacement).astype(int)
        out["breakout_follow_through"] = (
            (
                out["last_break_direction"].eq("UP")
                & out["close"].gt(out["last_broken_high"] + tolerance)
                & out["bars_since_bos"].between(1, 8, inclusive="both")
            )
            | (
                out["last_break_direction"].eq("DOWN")
                & out["close"].lt(out["last_broken_low"] - tolerance)
                & out["bars_since_bos"].between(1, 8, inclusive="both")
            )
        ).astype(int)

        swing_tolerance = avg_range * 0.35
        out["triple_top"] = (
            out["double_top"].eq(1)
            & out["swing_high"].rolling(20, min_periods=1).sum().ge(3)
            & out["high"].sub(out["prev_swing_high"]).abs().le(swing_tolerance)
        ).astype(int)
        out["triple_bottom"] = (
            out["double_bottom"].eq(1)
            & out["swing_low"].rolling(20, min_periods=1).sum().ge(3)
            & out["low"].sub(out["prev_swing_low"]).abs().le(swing_tolerance)
        ).astype(int)

        out["market_story"] = "NEUTRAL"
        out.loc[out["momentum_breakout"].eq(1) & out["bos_up"].eq(1), "market_story"] = "BULLISH_MOMENTUM_BREAKOUT"
        out.loc[out["momentum_breakout"].eq(1) & out["bos_down"].eq(1), "market_story"] = "BEARISH_MOMENTUM_BREAKOUT"
        out.loc[out["breakout_follow_through"].eq(1) & out["last_break_direction"].eq("UP"), "market_story"] = "BULLISH_CONTINUATION"
        out.loc[out["breakout_follow_through"].eq(1) & out["last_break_direction"].eq("DOWN"), "market_story"] = "BEARISH_CONTINUATION"
        out.loc[out["retest_broken_high"].eq(1), "market_story"] = "BULLISH_RETEST_ZONE"
        out.loc[out["retest_broken_low"].eq(1), "market_story"] = "BEARISH_RETEST_ZONE"
        out.loc[out["bullish_zone_rejection"].eq(1), "market_story"] = "BULLISH_RETEST_REJECTION"
        out.loc[out["bearish_zone_rejection"].eq(1), "market_story"] = "BEARISH_RETEST_REJECTION"
        out.loc[out["choch"].eq(1) & out["bullish_zone_rejection"].eq(1), "market_story"] = "BULLISH_REVERSAL_FROM_ZONE"
        out.loc[out["choch"].eq(1) & out["bearish_zone_rejection"].eq(1), "market_story"] = "BEARISH_REVERSAL_FROM_ZONE"
        out.loc[out["triple_top"].eq(1), "market_story"] = "BEARISH_TRIPLE_TOP_REVERSAL"
        out.loc[out["triple_bottom"].eq(1), "market_story"] = "BULLISH_TRIPLE_BOTTOM_REVERSAL"

        out["story_direction"] = "NEUTRAL"
        out.loc[out["market_story"].str.startswith("BULLISH"), "story_direction"] = "LONG"
        out.loc[out["market_story"].str.startswith("BEARISH"), "story_direction"] = "SHORT"
        out["story_confidence"] = 0.0
        out.loc[out["market_story"].str.contains("MOMENTUM_BREAKOUT|CONTINUATION", regex=True), "story_confidence"] += 45
        out.loc[out["market_story"].str.contains("RETEST_ZONE", regex=True), "story_confidence"] += 55
        out.loc[out["market_story"].str.contains("REJECTION|REVERSAL|TRIPLE", regex=True), "story_confidence"] += 70
        out.loc[out.get("order_block", pd.Series(0, index=index)).eq(1), "story_confidence"] += 8
        out.loc[out.get("fvg_zone", pd.Series(0, index=index)).eq(1), "story_confidence"] += 8
        out.loc[out["bos"].eq(1) | out["choch"].eq(1), "story_confidence"] += 10
        out["story_confidence"] = out["story_confidence"].clip(lower=0.0, upper=100.0)
        return out
