from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from strategy.decision_framework import (
    get_regime_behavior,
    score_trade,
    select_strategy_mode,
    should_enter_continuation_trade,
    should_enter_counter_trend_trade,
    should_enter_retracement_trade,
    should_enter_reversal_trade,
)


class FlowSystem:
    """Adaptive exploratory engine for broader FLOW setups."""

    def __init__(self, config: Any):
        self.config = config

    def generate_flow_setups(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._classify_smart_scalper_setups(out)
        out = self._apply_decision_framework(out)
        out["flow_direction"] = out["flow_direction"].replace("", np.nan)
        out.loc[out["flow_direction"].isna() & out["behavior_label"].eq("TREND_UP"), "flow_direction"] = "LONG"
        out.loc[out["flow_direction"].isna() & out["behavior_label"].eq("TREND_DOWN"), "flow_direction"] = "SHORT"
        out.loc[
            out["flow_direction"].isna()
            & out["pattern"].isin(["DOUBLE_BOTTOM", "BREAK_RETEST"])
            & out.get("bullish_reversal", pd.Series(0, index=out.index)).eq(1),
            "flow_direction",
        ] = "LONG"
        out.loc[
            out["flow_direction"].isna()
            & out["pattern"].isin(["DOUBLE_TOP", "BREAK_RETEST"])
            & out.get("bearish_reversal", pd.Series(0, index=out.index)).eq(1),
            "flow_direction",
        ] = "SHORT"
        story_direction = out.get("story_direction", pd.Series("NEUTRAL", index=out.index)).replace("", "NEUTRAL")
        story_signal = out.get("market_story", pd.Series("NEUTRAL", index=out.index)).astype(str)
        structure_intent = out.get("structure_trade_intent", pd.Series("WAIT", index=out.index)).astype(str)
        structure_confidence = out.get("structure_intent_confidence", pd.Series(0.0, index=out.index))
        out.loc[
            out["flow_direction"].isna()
            & story_direction.eq("LONG")
            & story_signal.str.contains("RETEST|CONTINUATION|REVERSAL|TRIPLE_BOTTOM|HEAD_SHOULDERS|MOMENTUM_BREAKOUT", regex=True),
            "flow_direction",
        ] = "LONG"
        out.loc[
            out["flow_direction"].isna()
            & story_direction.eq("SHORT")
            & story_signal.str.contains("RETEST|CONTINUATION|REVERSAL|TRIPLE_TOP|HEAD_SHOULDERS|MOMENTUM_BREAKOUT", regex=True),
            "flow_direction",
        ] = "SHORT"
        strong_story = out.get("story_confidence", pd.Series(0.0, index=out.index)).ge(70) & story_signal.str.contains(
            "RETEST_REJECTION|REVERSAL_FROM_ZONE|TRIPLE|HEAD_SHOULDERS", regex=True
        )
        out.loc[strong_story & story_direction.eq("LONG"), "flow_direction"] = "LONG"
        out.loc[strong_story & story_direction.eq("SHORT"), "flow_direction"] = "SHORT"
        out.loc[structure_confidence.ge(70) & structure_intent.str.endswith("_LONG"), "flow_direction"] = "LONG"
        out.loc[structure_confidence.ge(70) & structure_intent.str.endswith("_SHORT"), "flow_direction"] = "SHORT"
        out["flow_direction"] = out["flow_direction"].fillna("NEUTRAL")
        out = self._add_zone_indicator_confirmation(out)
        out = self._add_visual_zone_model(out)

        story_reversal = story_signal.str.contains("RETEST_REJECTION|REVERSAL_FROM_ZONE|TRIPLE|HEAD_SHOULDERS", regex=True) | structure_intent.str.contains("FAILED_RETEST|REVERSAL", regex=True)
        story_continuation = story_signal.str.contains("RETEST_ZONE|CONTINUATION|MOMENTUM_BREAKOUT", regex=True) | structure_intent.str.contains("RETEST_CONTINUATION|PULLBACK_RETEST|BREAKOUT", regex=True)
        out.loc[story_reversal & out["flow_trade_type"].eq("NONE"), "flow_trade_type"] = "ZONE_REVERSAL_REJECTION"
        out.loc[story_continuation & out["flow_trade_type"].eq("NONE"), "flow_trade_type"] = "STRUCTURE_RETEST_CONTINUATION"
        visual_zone = out.get("visual_zone_score", pd.Series(0.0, index=out.index)).ge(68)
        trap_reversal = (
            out.get("trap_reversal_score", pd.Series(0.0, index=out.index)).ge(55)
            & out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=out.index)).isin(["LONG", "SHORT"])
        )
        out.loc[visual_zone & out["flow_trade_type"].eq("NONE"), "flow_trade_type"] = "ZONE_REVERSAL_REJECTION"
        out.loc[trap_reversal & out["flow_trade_type"].eq("NONE"), "flow_trade_type"] = "ZONE_REVERSAL_REJECTION"
        out.loc[trap_reversal, "flow_direction"] = out.loc[trap_reversal, "trap_reversal_direction"]
        out.loc[story_reversal, "flow_counter_trend_allowed"] = 1
        out.loc[visual_zone, "flow_counter_trend_allowed"] = 1
        out.loc[trap_reversal, "flow_counter_trend_allowed"] = 1
        out.loc[story_reversal, "flow_atr_sl_multiplier"] = 0.5
        out.loc[story_continuation, "flow_atr_sl_multiplier"] = 0.45
        out.loc[visual_zone, "flow_atr_sl_multiplier"] = 0.48
        out.loc[story_reversal | story_continuation | visual_zone, "flow_rr_ratio"] = 2.0

        out["flow_score"] = 10
        out.loc[out["behavior_label"].isin(["RANGE", "BREAKOUT", "REVERSAL"]), "flow_score"] += 18
        out.loc[out["structure_state"].isin(["HL", "LH"]), "flow_score"] += 12
        out.loc[out["pattern"].isin(["DOUBLE_TOP", "DOUBLE_BOTTOM", "BREAK_RETEST", "CHOCH"]), "flow_score"] += 12
        out.loc[out["order_block"] == 1, "flow_score"] += 10
        out.loc[out["fvg_zone"] == 1, "flow_score"] += 10
        out.loc[out["supply_zone"] == 1, "flow_score"] += 8
        out.loc[out["demand_zone"] == 1, "flow_score"] += 8
        out.loc[out["session"].isin(["LONDON", "NEW_YORK"]), "flow_score"] += 6
        out.loc[out["behavior_confidence"] >= 55, "flow_score"] += 5
        out.loc[out["volatility"] == 1, "flow_score"] += 4
        out.loc[out["choppy"] == 1, "flow_score"] += 2
        out.loc[out.get("multi_tf_alignment_score", pd.Series(50.0, index=out.index)) >= 65, "flow_score"] += 10
        out.loc[out.get("breakout_quality", pd.Series(50.0, index=out.index)) >= 70, "flow_score"] += 8
        out.loc[out.get("continuation_strength", pd.Series(50.0, index=out.index)) >= 70, "flow_score"] += 6
        out.loc[out.get("liquidity_event", pd.Series("NONE", index=out.index)) == "CONFIRMED_SWEEP_REJECTION", "flow_score"] += 8
        out.loc[out["flow_trade_type"].eq("DEEP_PULLBACK_SCALP"), "flow_score"] += 12
        out.loc[out.get("market_story", pd.Series("NEUTRAL", index=out.index)).astype(str).str.contains("RETEST_REJECTION|REVERSAL_FROM_ZONE|TRIPLE|HEAD_SHOULDERS", regex=True), "flow_score"] += 14
        out.loc[out.get("market_story", pd.Series("NEUTRAL", index=out.index)).astype(str).str.contains("CONTINUATION|MOMENTUM_BREAKOUT", regex=True), "flow_score"] += 8
        out.loc[out.get("story_confidence", pd.Series(0.0, index=out.index)).ge(70), "flow_score"] += 8
        out.loc[structure_confidence.ge(70), "flow_score"] += 10
        out.loc[out.get("zone_indicator_score", pd.Series(0.0, index=out.index)).ge(60), "flow_score"] += 10
        out.loc[out.get("zone_indicator_score", pd.Series(0.0, index=out.index)).ge(80), "flow_score"] += 6
        out.loc[out.get("visual_zone_score", pd.Series(0.0, index=out.index)).ge(68), "flow_score"] += 12
        out.loc[out.get("visual_zone_score", pd.Series(0.0, index=out.index)).ge(82), "flow_score"] += 8
        out.loc[out.get("trap_reversal_score", pd.Series(0.0, index=out.index)).ge(55), "flow_score"] += 10
        out.loc[out.get("trap_reversal_score", pd.Series(0.0, index=out.index)).ge(75), "flow_score"] += 8
        out.loc[out.get("zone_indicator_conflict", pd.Series(0, index=out.index)).eq(1), "flow_score"] -= 18
        out.loc[out["flow_direction"].eq("LONG") & out.get("bullish_pattern_score", pd.Series(0, index=out.index)).gt(0), "flow_score"] += 8
        out.loc[out["flow_direction"].eq("SHORT") & out.get("bearish_pattern_score", pd.Series(0, index=out.index)).gt(0), "flow_score"] += 8
        out.loc[out["flow_direction"].eq("LONG") & out.get("bearish_reversal", pd.Series(0, index=out.index)).eq(1), "flow_score"] -= 15
        out.loc[out["flow_direction"].eq("SHORT") & out.get("bullish_reversal", pd.Series(0, index=out.index)).eq(1), "flow_score"] -= 15

        out = self._add_indicator_confirmation(out)
        out.loc[out["flow_indicator_confirmations"] >= 3, "flow_score"] += 10
        out.loc[out["flow_indicator_confirmations"] >= 4, "flow_score"] += 5
        out.loc[out["flow_indicator_conflict"].eq(1), "flow_score"] -= 25

        trap_reversal_aligned = (
            out.get("trap_reversal_score", pd.Series(0.0, index=out.index)).ge(55)
            & out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=out.index)).eq(out["flow_direction"])
        )
        out.loc[(out.get("fake_breakout", pd.Series(0, index=out.index)) == 1) & ~trap_reversal_aligned, "flow_score"] -= 35
        out.loc[out.get("trap_probability", pd.Series(0.0, index=out.index)) >= 70, "flow_score"] -= 20
        out.loc[out.get("htf_exhaustion", pd.Series(50.0, index=out.index)) >= 60, "flow_score"] -= 18
        out.loc[out.get("htf_liquidity_alignment", pd.Series(0, index=out.index)) < 0, "flow_score"] -= 18
        out.loc[out.get("multi_tf_alignment_score", pd.Series(50.0, index=out.index)) < 60, "flow_score"] -= 15
        out.loc[out.get("lifecycle_state", pd.Series("TREND_HEALTHY", index=out.index)).isin(["TREND_EXHAUSTING", "REVERSAL_WATCH"]), "flow_score"] -= 18
        out.loc[out.get("retracement_class", pd.Series("NON_TREND", index=out.index)) == "REVERSAL_WARNING", "flow_score"] -= 20
        out.loc[(out["behavior_label"] == "BREAKOUT") & (out.get("breakout_quality", pd.Series(50.0, index=out.index)) < 70), "flow_score"] -= 20
        out.loc[(out["behavior_label"] == "REVERSAL") & (out.get("confirmed_reversal", pd.Series(0, index=out.index)) == 0), "flow_score"] -= 20
        out["flow_score"] = out["flow_score"].clip(lower=0, upper=100)
        out["flow_score"] = out[["flow_score", "institutional_trade_score"]].max(axis=1).clip(lower=0, upper=100)

        min_flow_score = max(int(getattr(self.config.regime, "flow_min_confirm_score", 45)), 55)
        base_flow_allowed = (
            (out["flow_score"] >= min_flow_score)
            & (out["flow_direction"].isin(["LONG", "SHORT"]))
            & ((out.get("fake_breakout", pd.Series(0, index=out.index)) == 0) | trap_reversal_aligned)
            & (out.get("trap_probability", pd.Series(0.0, index=out.index)) < 75)
            & (out.get("multi_tf_alignment_score", pd.Series(50.0, index=out.index)) >= 60)
            & (~out.get("lifecycle_state", pd.Series("TREND_HEALTHY", index=out.index)).isin(["TREND_EXHAUSTING", "REVERSAL_WATCH"]))
        )
        framework_allowed = (
            (out["institutional_trade_score"] >= 60)
            & (out["flow_score"] >= min_flow_score - 5)
            & (out["flow_direction"].isin(["LONG", "SHORT"]))
            & (~out["strategy_mode"].eq("BOTH_PAUSED"))
            & (
                out["continuation_decision"].eq("VALID")
                | out["retracement_decision"].eq("ENTRY_OPPORTUNITY")
                | out["reversal_decision"].isin(["EARLY_WARNING", "CONFIRMED_REVERSAL"])
                | out["counter_trend_decision"].eq("ALLOWED")
            )
        )
        story_allowed = (
            out.get("story_confidence", pd.Series(0.0, index=out.index)).ge(70)
            & out["flow_direction"].isin(["LONG", "SHORT"])
            & (
                out.get("market_story", pd.Series("NEUTRAL", index=out.index)).astype(str).str.contains("RETEST_REJECTION|REVERSAL_FROM_ZONE|TRIPLE|HEAD_SHOULDERS|CONTINUATION", regex=True)
                | structure_intent.str.contains("RETEST_CONTINUATION|FAILED_RETEST|REVERSAL|BREAKOUT", regex=True)
            )
            & ((out.get("fake_breakout", pd.Series(0, index=out.index)) == 0) | trap_reversal_aligned)
            & (out.get("trap_probability", pd.Series(0.0, index=out.index)) < 75)
            & (out.get("zone_indicator_score", pd.Series(50.0, index=out.index)) >= 45)
        )
        visual_allowed = (
            out.get("visual_zone_score", pd.Series(0.0, index=out.index)).ge(78)
            & out["flow_direction"].isin(["LONG", "SHORT"])
            & out.get("visual_zone_direction", pd.Series("NEUTRAL", index=out.index)).eq(out["flow_direction"])
            & ((out.get("fake_breakout", pd.Series(0, index=out.index)) == 0) | trap_reversal_aligned)
            & (out.get("trap_probability", pd.Series(0.0, index=out.index)) < 75)
            & (out.get("zone_indicator_conflict", pd.Series(0, index=out.index)).eq(0) | out.get("visual_zone_score", pd.Series(0.0, index=out.index)).ge(86))
        )
        flow_allowed = base_flow_allowed | framework_allowed | story_allowed | visual_allowed

        continuation_flow = out["flow_trade_type"].isin(["NONE", "MOMENTUM_CONTINUATION", "MICRO_RETRACEMENT_REENTRY", "STRUCTURE_RETEST_CONTINUATION"])
        flow_allowed &= ~out["flow_indicator_conflict"].eq(1)
        flow_allowed &= (
            (out["flow_indicator_score"] >= 40)
            | (out.get("zone_indicator_score", pd.Series(0.0, index=out.index)) >= 55)
            | (~continuation_flow & out.get("confirmed_reversal", pd.Series(0, index=out.index)).eq(1))
        )

        counter_trend_against_m5 = (
            (out["behavior_label"].eq("TREND_UP") & out["flow_direction"].eq("SHORT"))
            | (out["behavior_label"].eq("TREND_DOWN") & out["flow_direction"].eq("LONG"))
        )
        counter_trend_without_reversal = counter_trend_against_m5 & (
            out["flow_counter_trend_allowed"].eq(0)
            | (
                out.get("confirmed_reversal", pd.Series(0, index=out.index)).eq(0)
                & ~out["counter_trend_decision"].eq("ALLOWED")
            )
        )
        counter_trend_without_reversal &= ~out["flow_trade_type"].isin(["DEEP_PULLBACK_SCALP", "ZONE_REVERSAL_REJECTION"])
        flow_allowed &= ~counter_trend_without_reversal

        trap_into_demand = (
            out["flow_direction"].eq("SHORT")
            & out.get("retest_zone_type", pd.Series("NONE", index=out.index)).isin(["SUPPORT_DEMAND", "BROKEN_HIGH"])
            & ~out.get("structure_trade_intent", pd.Series("WAIT", index=out.index)).astype(str).isin(["FAILED_RETEST_REVERSAL_SHORT", "REVERSAL_SHORT", "RETEST_CONTINUATION_SHORT", "BREAKOUT_SHORT"])
            & out.get("zone_reaction_direction", pd.Series("NEUTRAL", index=out.index)).ne("SHORT")
        )
        trap_into_supply = (
            out["flow_direction"].eq("LONG")
            & out.get("retest_zone_type", pd.Series("NONE", index=out.index)).isin(["RESISTANCE_SUPPLY", "BROKEN_LOW"])
            & ~out.get("structure_trade_intent", pd.Series("WAIT", index=out.index)).astype(str).isin(["FAILED_RETEST_REVERSAL_LONG", "REVERSAL_LONG", "RETEST_CONTINUATION_LONG", "BREAKOUT_LONG"])
            & out.get("zone_reaction_direction", pd.Series("NEUTRAL", index=out.index)).ne("LONG")
        )
        flow_allowed &= ~(trap_into_demand | trap_into_supply)

        out["flow_signal"] = np.where(flow_allowed, "FLOW", "")
        out["flow_notes"] = np.where(
            out["flow_signal"] == "FLOW",
            np.where(out["flow_trade_type"].ne("NONE"), out["flow_trade_type"], "filtered_exploratory"),
            "",
        )
        return out

    def _apply_decision_framework(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        records = out.to_dict(orient="records")
        out["continuation_decision"] = [should_enter_continuation_trade(row) for row in records]
        out["retracement_decision"] = [should_enter_retracement_trade(row) for row in records]
        out["reversal_decision"] = [should_enter_reversal_trade(row) for row in records]
        out["counter_trend_decision"] = [should_enter_counter_trend_trade(row) for row in records]
        out["strategy_mode"] = [select_strategy_mode(row) for row in records]
        out["regime_behavior"] = [get_regime_behavior(row) for row in records]
        out["institutional_trade_score"] = [score_trade(row) for row in records]
        return out

    def _classify_smart_scalper_setups(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        index = out.index

        out["flow_trade_type"] = "NONE"
        out["flow_direction"] = ""
        out["flow_atr_sl_multiplier"] = 0.5
        out["flow_rr_ratio"] = 1.5
        out["flow_signal_expiry_minutes"] = 3
        out["flow_counter_trend_allowed"] = 0
        out["flow_max_open_trades"] = 3

        candle_body = (out["close"] - out["open"]).abs()
        candle_range = (out["high"] - out["low"]).replace(0, np.nan)
        upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
        lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
        impulse = out["close"].diff().abs()
        avg_impulse = impulse.rolling(20, min_periods=5).mean()
        impulse_extension = impulse / avg_impulse.replace(0, np.nan)
        rsi14 = self._rsi(out["close"])
        previous_body = candle_body.shift(1)
        previous_body = previous_body.where(previous_body.notna(), candle_body)

        trend_up = out["behavior_label"].eq("TREND_UP")
        trend_down = out["behavior_label"].eq("TREND_DOWN")
        trend = trend_up | trend_down
        bullish_close = out["close"] > out["open"]
        bearish_close = out["close"] < out["open"]
        bullish_rejection = bullish_close & (lower_wick >= candle_body * 0.8)
        bearish_rejection = bearish_close & (upper_wick >= candle_body * 0.8)
        bullish_engulf = bullish_close & (out["close"] > out["open"].shift(1)) & (out["open"] <= out["close"].shift(1))
        bearish_engulf = bearish_close & (out["close"] < out["open"].shift(1)) & (out["open"] >= out["close"].shift(1))
        volume_confirm = out.get("tick_volume", pd.Series(0.0, index=index)) >= out.get("tick_volume", pd.Series(0.0, index=index)).rolling(20, min_periods=1).mean()
        momentum_confirm = (
            (trend_up & out.get("momentum", pd.Series(0.0, index=index)).gt(0))
            | (trend_down & out.get("momentum", pd.Series(0.0, index=index)).lt(0))
        )
        retracement = out.get("fib_retracement_pct", pd.Series(np.nan, index=index))

        momentum_continuation = (
            trend
            & retracement.between(20.0, 38.0, inclusive="both")
            & (out.get("bos", pd.Series(0, index=index)) == 0)
            & (out.get("choch", pd.Series(0, index=index)) == 0)
            & ((trend_up & bullish_close) | (trend_down & bearish_close))
        )
        reentry = (
            trend
            & retracement.between(38.0, 50.0, inclusive="both")
            & ((trend_up & (bullish_rejection | bullish_engulf)) | (trend_down & (bearish_rejection | bearish_engulf)))
            & (volume_confirm | momentum_confirm)
        )
        exhaustion_fade = (
            trend
            & impulse_extension.gt(1.5)
            & (
                (trend_up & rsi14.gt(78) & (upper_wick >= candle_body))
                | (trend_down & rsi14.lt(22) & (lower_wick >= candle_body))
            )
            & (candle_body < previous_body)
        )
        early_reversal = (
            trend
            & (out.get("choch", pd.Series(0, index=index)) == 1)
            & (out.get("bos", pd.Series(0, index=index)) == 0)
            & retracement.between(50.0, 78.6, inclusive="both")
            & ((trend_up & bearish_rejection) | (trend_down & bullish_rejection))
        )
        deep_pullback_scalp = (
            trend
            & retracement.between(50.0, 78.6, inclusive="both")
            & (
                (trend_down & (bullish_rejection | bullish_engulf) & rsi14.between(25.0, 65.0, inclusive="both"))
                | (trend_up & (bearish_rejection | bearish_engulf) & rsi14.between(35.0, 75.0, inclusive="both"))
            )
            & volume_confirm
        )

        self._assign_setup(out, momentum_continuation, "MOMENTUM_CONTINUATION", trend_up, trend_down, 0.5, 2.0, False)
        self._assign_setup(out, reentry & out["flow_trade_type"].eq("NONE"), "MICRO_RETRACEMENT_REENTRY", trend_up, trend_down, 0.4, 2.0, False)
        self._assign_setup(out, deep_pullback_scalp & out["flow_trade_type"].eq("NONE"), "DEEP_PULLBACK_SCALP", trend_down, trend_up, 0.45, 1.5, True)
        self._assign_setup(out, exhaustion_fade & out["flow_trade_type"].eq("NONE"), "EXHAUSTION_FADE", trend_down, trend_up, 0.3, 2.0, True)
        self._assign_setup(out, early_reversal & out["flow_trade_type"].eq("NONE"), "EARLY_REVERSAL_ENTRY", trend_down, trend_up, 0.6, 2.0, True)
        return out

    def _assign_setup(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        name: str,
        long_mask: pd.Series,
        short_mask: pd.Series,
        sl_multiplier: float,
        rr_ratio: float,
        counter_trend: bool,
    ) -> None:
        df.loc[mask, "flow_trade_type"] = name
        df.loc[mask & long_mask, "flow_direction"] = "LONG"
        df.loc[mask & short_mask, "flow_direction"] = "SHORT"
        df.loc[mask, "flow_atr_sl_multiplier"] = sl_multiplier
        df.loc[mask, "flow_rr_ratio"] = max(1.5, rr_ratio)
        df.loc[mask, "flow_counter_trend_allowed"] = int(counter_trend)

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50.0)

    def _add_zone_indicator_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        index = out.index

        if "rsi14" not in out.columns:
            out["rsi14"] = self._rsi(out["close"])

        volume = out.get("tick_volume", out.get("volume", pd.Series(1.0, index=index))).fillna(1.0)
        volume_avg = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
        out["volume_avg_20"] = volume_avg.fillna(1.0)
        out["zone_volume_ratio"] = (volume / volume_avg).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["zone_volume_spike"] = out["zone_volume_ratio"].ge(1.25).astype(int)

        close = out["close"]
        open_ = out["open"]
        bullish_candle = close > open_
        bearish_candle = close < open_
        momentum = out.get("momentum", close.diff()).fillna(0.0)
        momentum_3 = momentum.rolling(3, min_periods=1).mean()
        rsi = out["rsi14"].fillna(50.0)
        rsi_slope = rsi.diff().fillna(0.0)

        macd_hist = out.get("macd_histogram", pd.Series(0.0, index=index)).fillna(0.0)
        macd_slope = out.get("macd_slope", pd.Series(0.0, index=index)).fillna(0.0)
        macd_cross = out.get("macd_crossover", pd.Series(0, index=index)).fillna(0)
        plus_di = out.get("adx_plus_di", pd.Series(0.0, index=index)).fillna(0.0)
        minus_di = out.get("adx_minus_di", pd.Series(0.0, index=index)).fillna(0.0)
        adx = out.get("adx", pd.Series(0.0, index=index)).fillna(0.0)
        stoch_k = out.get("stoch_k", pd.Series(50.0, index=index)).fillna(50.0)
        stoch_d = out.get("stoch_d", pd.Series(50.0, index=index)).fillna(50.0)
        stoch_bull = out.get("stoch_bullish_cross", pd.Series(0, index=index)).fillna(0)
        stoch_bear = out.get("stoch_bearish_cross", pd.Series(0, index=index)).fillna(0)
        bb_touch_lower = out.get("bb_touch_lower", pd.Series(0, index=index)).fillna(0)
        bb_touch_upper = out.get("bb_touch_upper", pd.Series(0, index=index)).fillna(0)

        zone_active = (
            out.get("retest_zone_type", pd.Series("NONE", index=index)).ne("NONE")
            | out.get("structure_retest_active", pd.Series(0, index=index)).eq(1)
            | out.get("zone_confluence_score", pd.Series(0.0, index=index)).gt(0)
            | out.get("market_story", pd.Series("NEUTRAL", index=index)).astype(str).str.contains("RETEST|REVERSAL|TRIPLE|HEAD_SHOULDERS", regex=True)
        )

        bullish_count = (
            bullish_candle.astype(int)
            + momentum_3.gt(0).astype(int)
            + (macd_hist.gt(0) | macd_slope.gt(0) | macd_cross.eq(1)).astype(int)
            + ((plus_di.gt(minus_di)) & adx.ge(15)).astype(int)
            + ((stoch_k.gt(stoch_d)) | stoch_bull.eq(1)).astype(int)
            + ((rsi.lt(45) & rsi_slope.gt(0)) | rsi.between(45, 62, inclusive="both")).astype(int)
            + bb_touch_lower.eq(1).astype(int)
        )
        bearish_count = (
            bearish_candle.astype(int)
            + momentum_3.lt(0).astype(int)
            + (macd_hist.lt(0) | macd_slope.lt(0) | macd_cross.eq(-1)).astype(int)
            + ((minus_di.gt(plus_di)) & adx.ge(15)).astype(int)
            + ((stoch_k.lt(stoch_d)) | stoch_bear.eq(1)).astype(int)
            + ((rsi.gt(55) & rsi_slope.lt(0)) | rsi.between(38, 55, inclusive="both")).astype(int)
            + bb_touch_upper.eq(1).astype(int)
        )

        volume_bonus = out["zone_volume_spike"] * (
            bullish_candle.astype(int).where(out["flow_direction"].eq("LONG"), bearish_candle.astype(int))
        )
        out["zone_bull_indicator_count"] = bullish_count.where(zone_active, 0)
        out["zone_bear_indicator_count"] = bearish_count.where(zone_active, 0)
        out["zone_indicator_direction"] = "NEUTRAL"
        out.loc[zone_active & bullish_count.gt(bearish_count + 1), "zone_indicator_direction"] = "LONG"
        out.loc[zone_active & bearish_count.gt(bullish_count + 1), "zone_indicator_direction"] = "SHORT"

        aligned_count = np.select(
            [out["flow_direction"].eq("LONG"), out["flow_direction"].eq("SHORT")],
            [bullish_count, bearish_count],
            default=0,
        )
        opposing_count = np.select(
            [out["flow_direction"].eq("LONG"), out["flow_direction"].eq("SHORT")],
            [bearish_count, bullish_count],
            default=0,
        )
        out["zone_indicator_score"] = (
            pd.Series(aligned_count, index=index).astype(float) * 12.0
            + out.get("zone_confluence_score", pd.Series(0.0, index=index)).fillna(0.0) * 0.25
            + volume_bonus.astype(float) * 10.0
        ).where(zone_active, 0.0).clip(0.0, 100.0)
        out["zone_indicator_conflict"] = (
            zone_active
            & out["flow_direction"].isin(["LONG", "SHORT"])
            & pd.Series(opposing_count, index=index).ge(pd.Series(aligned_count, index=index) + 2)
        ).astype(int)
        return out

    def _add_visual_zone_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score the screenshot-style red/blue circle zones: level, reaction, confirmation."""
        out = df.copy()
        index = out.index

        close = out["close"]
        open_ = out["open"]
        candle_range = (out["high"] - out["low"]).replace(0, np.nan)
        body = (close - open_).abs()
        upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
        lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
        bullish_candle = close > open_
        bearish_candle = close < open_
        bullish_rejection = bullish_candle & (lower_wick >= body * 0.7)
        bearish_rejection = bearish_candle & (upper_wick >= body * 0.7)

        rolling_low = out["low"].rolling(80, min_periods=10).min()
        rolling_high = out["high"].rolling(80, min_periods=10).max()
        range_position = ((close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)).clip(0, 1).fillna(0.5)

        retest_type = out.get("retest_zone_type", pd.Series("NONE", index=index)).astype(str)
        story = out.get("market_story", pd.Series("NEUTRAL", index=index)).astype(str)
        intent = out.get("structure_trade_intent", pd.Series("WAIT", index=index)).astype(str)
        reaction = out.get("zone_reaction_direction", pd.Series("NEUTRAL", index=index)).astype(str)
        indicator_dir = out.get("zone_indicator_direction", pd.Series("NEUTRAL", index=index)).astype(str)
        indicator_score = out.get("zone_indicator_score", pd.Series(0.0, index=index)).fillna(0.0)
        volume_ratio = out.get("zone_volume_ratio", pd.Series(1.0, index=index)).fillna(1.0)
        confluence = out.get("zone_confluence_score", pd.Series(0.0, index=index)).fillna(0.0)

        demand_context = (
            retest_type.isin(["SUPPORT_DEMAND", "BROKEN_HIGH", "ORDER_BLOCK", "FVG"])
            | out.get("demand_zone", pd.Series(0, index=index)).eq(1)
            | out.get("is_support", pd.Series(0, index=index)).eq(1)
            | story.str.startswith("BULLISH")
            | intent.str.endswith("_LONG")
        )
        supply_context = (
            retest_type.isin(["RESISTANCE_SUPPLY", "BROKEN_LOW", "ORDER_BLOCK", "FVG"])
            | out.get("supply_zone", pd.Series(0, index=index)).eq(1)
            | out.get("is_resistance", pd.Series(0, index=index)).eq(1)
            | story.str.startswith("BEARISH")
            | intent.str.endswith("_SHORT")
        )

        buy_pattern = (
            out.get("bullish_reversal", pd.Series(0, index=index)).eq(1)
            | out.get("double_bottom", pd.Series(0, index=index)).eq(1)
            | out.get("triple_bottom", pd.Series(0, index=index)).eq(1)
            | out.get("head_shoulders_bottom", pd.Series(0, index=index)).eq(1)
            | out.get("bullish_zone_rejection", pd.Series(0, index=index)).eq(1)
        )
        sell_pattern = (
            out.get("bearish_reversal", pd.Series(0, index=index)).eq(1)
            | out.get("double_top", pd.Series(0, index=index)).eq(1)
            | out.get("triple_top", pd.Series(0, index=index)).eq(1)
            | out.get("head_shoulders_top", pd.Series(0, index=index)).eq(1)
            | out.get("bearish_zone_rejection", pd.Series(0, index=index)).eq(1)
        )

        buy_score = (
            demand_context.astype(float) * 24.0
            + confluence * 0.28
            + (range_position <= 0.38).astype(float) * 10.0
            + (bullish_rejection | buy_pattern | reaction.eq("LONG")).astype(float) * 22.0
            + indicator_dir.eq("LONG").astype(float) * 12.0
            + indicator_score.where(indicator_dir.eq("LONG"), 0.0) * 0.22
            + volume_ratio.ge(1.20).astype(float) * 8.0
            + out.get("liquidity_event", pd.Series("NONE", index=index)).astype(str).str.contains("SWEEP|REJECTION", regex=True).astype(float) * 8.0
            + out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=index)).eq("LONG").astype(float) * 10.0
            + out.get("trap_reversal_score", pd.Series(0.0, index=index)).where(out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=index)).eq("LONG"), 0.0) * 0.18
        ).clip(0.0, 100.0)
        sell_score = (
            supply_context.astype(float) * 24.0
            + confluence * 0.28
            + (range_position >= 0.62).astype(float) * 10.0
            + (bearish_rejection | sell_pattern | reaction.eq("SHORT")).astype(float) * 22.0
            + indicator_dir.eq("SHORT").astype(float) * 12.0
            + indicator_score.where(indicator_dir.eq("SHORT"), 0.0) * 0.22
            + volume_ratio.ge(1.20).astype(float) * 8.0
            + out.get("liquidity_event", pd.Series("NONE", index=index)).astype(str).str.contains("SWEEP|REJECTION", regex=True).astype(float) * 8.0
            + out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=index)).eq("SHORT").astype(float) * 10.0
            + out.get("trap_reversal_score", pd.Series(0.0, index=index)).where(out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=index)).eq("SHORT"), 0.0) * 0.18
        ).clip(0.0, 100.0)

        out["visual_buy_zone_score"] = buy_score
        out["visual_sell_zone_score"] = sell_score
        out["visual_zone_direction"] = "NEUTRAL"
        out.loc[buy_score.ge(60) & buy_score.gt(sell_score + 8), "visual_zone_direction"] = "LONG"
        out.loc[sell_score.ge(60) & sell_score.gt(buy_score + 8), "visual_zone_direction"] = "SHORT"
        out["visual_zone_score"] = np.select(
            [out["visual_zone_direction"].eq("LONG"), out["visual_zone_direction"].eq("SHORT")],
            [buy_score, sell_score],
            default=np.maximum(buy_score, sell_score),
        ).astype(float).clip(0.0, 100.0)

        out["visual_zone_type"] = "NONE"
        out.loc[out["visual_zone_direction"].eq("LONG") & retest_type.eq("BROKEN_HIGH"), "visual_zone_type"] = "BROKEN_HIGH_BUY_RETEST"
        out.loc[out["visual_zone_direction"].eq("LONG") & out["visual_zone_type"].eq("NONE") & demand_context, "visual_zone_type"] = "DEMAND_BUY_REJECTION"
        out.loc[out["visual_zone_direction"].eq("SHORT") & retest_type.eq("BROKEN_LOW"), "visual_zone_type"] = "BROKEN_LOW_SELL_RETEST"
        out.loc[out["visual_zone_direction"].eq("SHORT") & out["visual_zone_type"].eq("NONE") & supply_context, "visual_zone_type"] = "SUPPLY_SELL_REJECTION"
        out.loc[
            out["visual_zone_direction"].eq(out.get("trap_reversal_direction", pd.Series("NEUTRAL", index=index)))
            & out.get("trap_reversal_score", pd.Series(0.0, index=index)).ge(55),
            "visual_zone_type",
        ] = "LIQUIDITY_TRAP_REVERSAL"
        out.loc[out["visual_zone_direction"].ne("NEUTRAL") & out["visual_zone_type"].eq("NONE"), "visual_zone_type"] = "ZONE_REACTION"

        high_confidence = out["visual_zone_score"].ge(78)
        out.loc[high_confidence & out["visual_zone_direction"].eq("LONG"), "flow_direction"] = "LONG"
        out.loc[high_confidence & out["visual_zone_direction"].eq("SHORT"), "flow_direction"] = "SHORT"
        out["visual_zone_reason"] = ""
        out.loc[high_confidence, "visual_zone_reason"] = (
            out.loc[high_confidence, "visual_zone_type"].astype(str)
            + "|score="
            + out.loc[high_confidence, "visual_zone_score"].round(0).astype(int).astype(str)
            + "|ind="
            + out.loc[high_confidence, "zone_indicator_direction"].astype(str)
        )
        return out

    def score_experimental_patterns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        out = dataframe.copy()
        out["experiment_score"] = 0
        out.loc[out["pattern"] == "CHOCH", "experiment_score"] += 15
        out.loc[out["pattern"].isin(["DOUBLE_TOP", "DOUBLE_BOTTOM"]), "experiment_score"] += 10
        out.loc[out["fvg_zone"] == 1, "experiment_score"] += 8
        return out

    def _add_indicator_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        index = out.index

        close = out.get("close", pd.Series(0.0, index=index))
        macd_hist = out.get("macd_histogram", pd.Series(0.0, index=index)).fillna(0.0)
        macd_slope = out.get("macd_slope", pd.Series(0.0, index=index)).fillna(0.0)
        plus_di = out.get("adx_plus_di", pd.Series(0.0, index=index)).fillna(0.0)
        minus_di = out.get("adx_minus_di", pd.Series(0.0, index=index)).fillna(0.0)
        adx = out.get("adx", pd.Series(0.0, index=index)).fillna(0.0)
        stoch_k = out.get("stoch_k", pd.Series(50.0, index=index)).fillna(50.0)
        stoch_d = out.get("stoch_d", pd.Series(50.0, index=index)).fillna(50.0)
        bb_middle = out.get("bb_middle", close).fillna(close)

        bull_count = (
            macd_hist.gt(0).astype(int)
            + macd_slope.gt(0).astype(int)
            + (plus_di.gt(minus_di) & adx.ge(15)).astype(int)
            + stoch_k.ge(stoch_d).astype(int)
            + close.ge(bb_middle).astype(int)
        )
        bear_count = (
            macd_hist.lt(0).astype(int)
            + macd_slope.lt(0).astype(int)
            + (minus_di.gt(plus_di) & adx.ge(15)).astype(int)
            + stoch_k.le(stoch_d).astype(int)
            + close.le(bb_middle).astype(int)
        )

        out["flow_bull_indicator_confirmations"] = bull_count
        out["flow_bear_indicator_confirmations"] = bear_count
        out["flow_indicator_confirmations"] = np.select(
            [out["flow_direction"].eq("LONG"), out["flow_direction"].eq("SHORT")],
            [bull_count, bear_count],
            default=0,
        )
        out["flow_indicator_score"] = (out["flow_indicator_confirmations"] * 20).clip(lower=0, upper=100)
        out["flow_indicator_conflict"] = np.select(
            [
                out["flow_direction"].eq("LONG") & bear_count.ge(4) & bull_count.le(2),
                out["flow_direction"].eq("SHORT") & bull_count.ge(4) & bear_count.le(2),
            ],
            [1, 1],
            default=0,
        )
        return out
