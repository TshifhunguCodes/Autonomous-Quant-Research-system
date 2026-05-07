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
        out.loc[out.get("fake_breakout", pd.Series(0, index=out.index)) == 1, "flow_score"] -= 35
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

        min_flow_score = max(int(getattr(self.config.regime, "flow_min_confirm_score", 45)), 45)
        base_flow_allowed = (
            (out["flow_score"] >= min_flow_score)
            & (out.get("fake_breakout", pd.Series(0, index=out.index)) == 0)
            & (out.get("trap_probability", pd.Series(0.0, index=out.index)) < 75)
            & (out.get("multi_tf_alignment_score", pd.Series(50.0, index=out.index)) >= 60)
            & (~out.get("lifecycle_state", pd.Series("TREND_HEALTHY", index=out.index)).isin(["TREND_EXHAUSTING", "REVERSAL_WATCH"]))
        )
        framework_allowed = (
            (out["institutional_trade_score"] >= 45)
            & (~out["strategy_mode"].eq("BOTH_PAUSED"))
            & (
                out["continuation_decision"].eq("VALID")
                | out["retracement_decision"].eq("ENTRY_OPPORTUNITY")
                | out["reversal_decision"].isin(["EARLY_WARNING", "CONFIRMED_REVERSAL"])
                | out["counter_trend_decision"].eq("ALLOWED")
            )
        )
        flow_allowed = base_flow_allowed | framework_allowed

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

        self._assign_setup(out, momentum_continuation, "MOMENTUM_CONTINUATION", trend_up, trend_down, 0.5, 2.0, False)
        self._assign_setup(out, reentry & out["flow_trade_type"].eq("NONE"), "MICRO_RETRACEMENT_REENTRY", trend_up, trend_down, 0.4, 2.0, False)
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

    def score_experimental_patterns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        out = dataframe.copy()
        out["experiment_score"] = 0
        out.loc[out["pattern"] == "CHOCH", "experiment_score"] += 15
        out.loc[out["pattern"].isin(["DOUBLE_TOP", "DOUBLE_BOTTOM"]), "experiment_score"] += 10
        out.loc[out["fvg_zone"] == 1, "experiment_score"] += 8
        return out
