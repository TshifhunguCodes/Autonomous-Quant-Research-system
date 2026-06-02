from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class LiquidityEngine:
    """Detects lightweight liquidity events around sweeps, traps, and failed breakouts."""

    def __init__(self, config: Any):
        self.config = config

    def classify_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        candle_body = (out["close"] - out["open"]).abs()
        candle_range = (out["high"] - out["low"]).replace(0, np.nan)
        upper_wick = (out["high"] - out[["open", "close"]].max(axis=1)).clip(lower=0.0)
        lower_wick = (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0.0)
        avg_range = candle_range.rolling(20, min_periods=1).mean().fillna(candle_range)
        atr_mean = out["atr14"].rolling(20, min_periods=1).mean()

        wick_rejection = ((upper_wick > candle_body * 1.2) | (lower_wick > candle_body * 1.2)).fillna(False)
        volatility_spike = ((candle_range >= avg_range * 1.35) | (out["atr14"] >= atr_mean * 1.2)).fillna(False)
        breakout_attempt = (
            out["bos"].fillna(0).astype(int).eq(1)
            | out["bos_up"].fillna(0).astype(int).eq(1)
            | out["bos_down"].fillna(0).astype(int).eq(1)
            | out["break_retest"].fillna(0).astype(int).eq(1)
        )
        follow_through = (
            (out["close"] >= out["high"].shift(1))
            | (out["close"] <= out["low"].shift(1))
            | (out["range"] >= avg_range)
        ).fillna(False)

        trend_up = out["behavior_label"].eq("TREND_UP")
        trend_down = out["behavior_label"].eq("TREND_DOWN")
        reversal_watch = out["lifecycle_state"].isin(["REVERSAL_WATCH", "REVERSAL_CONFIRMED"])
        exhaustion = out["exhaustion_score"].fillna(0.0)
        continuation_failure = out["failed_continuation"].fillna(0).astype(int).eq(1) | out["failed_bos"].fillna(0).astype(int).eq(1)

        sweep_high = out.get("sweep_high", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
        sweep_low = out.get("sweep_low", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
        structure_rejection = (
            (trend_up & upper_wick.gt(lower_wick) & wick_rejection)
            | (trend_down & lower_wick.gt(upper_wick) & wick_rejection)
            | reversal_watch
        )

        out["liquidity_sweep"] = (sweep_high | sweep_low | (volatility_spike & wick_rejection & structure_rejection)).astype(int)
        out["stop_hunt_detected"] = (
            out["liquidity_sweep"].eq(1)
            & volatility_spike
            & wick_rejection
            & exhaustion.ge(55)
        ).astype(int)

        out["fake_breakout"] = (
            breakout_attempt
            & (~follow_through)
            & wick_rejection
            & (continuation_failure | structure_rejection)
        ).astype(int)

        failed_bos_continuation = breakout_attempt & continuation_failure & (~follow_through)
        breakout_rejection = breakout_attempt & wick_rejection & (~follow_through)
        trap_breakout = out["fake_breakout"].eq(1) & volatility_spike & exhaustion.ge(60)
        failed_up_break = (
            out.get("bos_up", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
            & upper_wick.gt(lower_wick)
            & out["close"].le(out.get("prev_swing_high", out["high"].shift(1)))
        )
        failed_down_break = (
            out.get("bos_down", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
            & lower_wick.gt(upper_wick)
            & out["close"].ge(out.get("prev_swing_low", out["low"].shift(1)))
        )
        fake_up_reversal = (
            out["fake_breakout"].eq(1)
            & upper_wick.gt(lower_wick * 1.15)
            & (
                out.get("bos_up", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
                | sweep_high
                | out.get("supply_zone", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
                | out.get("is_resistance", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
            )
        )
        fake_down_reversal = (
            out["fake_breakout"].eq(1)
            & lower_wick.gt(upper_wick * 1.15)
            & (
                out.get("bos_down", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
                | sweep_low
                | out.get("demand_zone", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
                | out.get("is_support", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1)
            )
        )
        double_top_trap = out.get("double_top", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1) & (
            sweep_high | upper_wick.gt(candle_body)
        )
        double_bottom_trap = out.get("double_bottom", pd.Series(0, index=out.index)).fillna(0).astype(int).eq(1) & (
            sweep_low | lower_wick.gt(candle_body)
        )
        confirmed_sweep_rejection = (
            out["liquidity_sweep"].eq(1)
            & wick_rejection
            & reversal_watch
            & out["confirmed_reversal"].fillna(0).astype(int).eq(1)
        )

        breakout_quality = pd.Series(45.0, index=out.index)
        breakout_quality += breakout_attempt.astype(float) * 10.0
        breakout_quality += follow_through.astype(float) * 20.0
        breakout_quality += out["continuation_strength"].fillna(0.0) * 0.2
        breakout_quality -= out["fake_breakout"].astype(float) * 30.0
        breakout_quality -= breakout_rejection.astype(float) * 15.0
        breakout_quality -= exhaustion * 0.15
        out["breakout_quality"] = breakout_quality.clip(lower=0.0, upper=100.0)

        trap_probability = pd.Series(20.0, index=out.index)
        trap_probability += out["fake_breakout"].astype(float) * 30.0
        trap_probability += trap_breakout.astype(float) * 20.0
        trap_probability += out["stop_hunt_detected"].astype(float) * 15.0
        trap_probability += exhaustion * 0.2
        trap_probability -= follow_through.astype(float) * 15.0
        out["trap_probability"] = trap_probability.clip(lower=0.0, upper=100.0)

        out["trap_reversal_direction"] = "NEUTRAL"
        out.loc[sweep_low | failed_down_break | fake_down_reversal | double_bottom_trap, "trap_reversal_direction"] = "LONG"
        out.loc[sweep_high | failed_up_break | fake_up_reversal | double_top_trap, "trap_reversal_direction"] = "SHORT"
        out["trap_pattern"] = "NONE"
        out.loc[failed_up_break, "trap_pattern"] = "FAILED_UP_BREAKOUT"
        out.loc[failed_down_break, "trap_pattern"] = "FAILED_DOWN_BREAKOUT"
        out.loc[fake_up_reversal, "trap_pattern"] = "FAKE_UP_BREAKOUT_REJECTION"
        out.loc[fake_down_reversal, "trap_pattern"] = "FAKE_DOWN_BREAKOUT_REJECTION"
        out.loc[double_top_trap, "trap_pattern"] = "DOUBLE_TOP_SWEEP"
        out.loc[double_bottom_trap, "trap_pattern"] = "DOUBLE_BOTTOM_SWEEP"
        out["trap_reversal_score"] = (
            out["liquidity_sweep"].astype(float) * 20.0
            + out["fake_breakout"].astype(float) * 25.0
            + trap_breakout.astype(float) * 20.0
            + (double_top_trap | double_bottom_trap).astype(float) * 18.0
            + (failed_up_break | failed_down_break | fake_up_reversal | fake_down_reversal).astype(float) * 18.0
            + wick_rejection.astype(float) * 10.0
            + volatility_spike.astype(float) * 8.0
            + exhaustion.clip(0.0, 100.0) * 0.10
            - follow_through.astype(float) * 15.0
        ).clip(lower=0.0, upper=100.0)

        out["liquidity_event"] = "NONE"
        out.loc[failed_bos_continuation, "liquidity_event"] = "FAILED_BOS_CONTINUATION"
        out.loc[breakout_rejection, "liquidity_event"] = "BREAKOUT_REJECTION"
        out.loc[out["liquidity_sweep"].eq(1), "liquidity_event"] = "LIQUIDITY_SWEEP"
        out.loc[out["stop_hunt_detected"].eq(1), "liquidity_event"] = "STOP_HUNT"
        out.loc[out["fake_breakout"].eq(1), "liquidity_event"] = "FAKE_BREAKOUT"
        out.loc[trap_breakout, "liquidity_event"] = "TRAP_BREAKOUT"
        out.loc[double_top_trap | double_bottom_trap, "liquidity_event"] = "PATTERN_LIQUIDITY_TRAP"
        out.loc[failed_up_break | failed_down_break | fake_up_reversal | fake_down_reversal, "liquidity_event"] = "FAILED_BREAKOUT_REVERSAL"
        out.loc[confirmed_sweep_rejection, "liquidity_event"] = "CONFIRMED_SWEEP_REJECTION"

        return out
