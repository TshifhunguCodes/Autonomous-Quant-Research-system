from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MTFContextEngine:
    """Builds lightweight higher-timeframe alignment context from H1 and derived H4 candles."""

    def __init__(self, config: Any):
        self.config = config

    def classify_context(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "direction" not in out.columns:
            out["direction"] = "NEUTRAL"
            out.loc[out["behavior_label"].eq("TREND_UP"), "direction"] = "LONG"
            out.loc[out["behavior_label"].eq("TREND_DOWN"), "direction"] = "SHORT"
        h1 = self._load_h1()
        if h1.empty:
            out["htf_bias"] = "NEUTRAL"
            out["htf_lifecycle"] = "UNKNOWN"
            out["htf_exhaustion"] = 50.0
            out["htf_liquidity_alignment"] = 0
            out["multi_tf_alignment_score"] = 50.0
            return out

        h1_context = self._build_h1_context(h1)
        h4_context = self._build_h4_context(h1)

        merged = pd.merge_asof(
            out.sort_values("time"),
            h1_context.sort_values("time"),
            on="time",
            direction="backward",
        )
        merged = pd.merge_asof(
            merged.sort_values("time"),
            h4_context.sort_values("time"),
            on="time",
            direction="backward",
        )

        merged["htf_exhaustion"] = merged["h4_exhaustion"].fillna(merged["h1_exhaustion"]).fillna(50.0)
        merged["htf_lifecycle"] = merged["h1_lifecycle"].fillna("UNKNOWN")

        long_aligned = (
            merged["direction"].eq("LONG")
            & merged["h1_bias"].eq("BULLISH")
            & ~merged["h1_supply_rejection"].fillna(0).astype(bool)
        )
        short_aligned = (
            merged["direction"].eq("SHORT")
            & merged["h1_bias"].eq("BEARISH")
            & ~merged["h1_demand_rejection"].fillna(0).astype(bool)
        )
        merged["htf_bias"] = "NEUTRAL"
        merged.loc[merged["h1_bias"].eq("BULLISH"), "htf_bias"] = "BULLISH"
        merged.loc[merged["h1_bias"].eq("BEARISH"), "htf_bias"] = "BEARISH"

        merged["htf_liquidity_alignment"] = 0
        merged.loc[long_aligned | short_aligned, "htf_liquidity_alignment"] = 1
        merged.loc[
            merged["h1_supply_rejection"].fillna(0).astype(bool) | merged["h1_demand_rejection"].fillna(0).astype(bool),
            "htf_liquidity_alignment",
        ] = -1

        alignment_score = pd.Series(45.0, index=merged.index)
        alignment_score += (long_aligned | short_aligned).astype(float) * 25.0
        alignment_score += merged["h1_bos_bias_match"].fillna(0).astype(float) * 10.0
        alignment_score += merged["h1_breakout_supportive"].fillna(0).astype(float) * 10.0
        alignment_score -= merged["htf_exhaustion"] * 0.2
        alignment_score -= merged["h4_reversal_risk"].fillna(0.0) * 0.2
        alignment_score -= (
            merged["h1_supply_rejection"].fillna(0).astype(float) * 15.0
            + merged["h1_demand_rejection"].fillna(0).astype(float) * 15.0
        )
        merged["multi_tf_alignment_score"] = alignment_score.clip(lower=0.0, upper=100.0)

        return merged

    def _load_h1(self) -> pd.DataFrame:
        source = self.config.base.paths.clean_h1
        if not source.exists():
            source = self.config.base.paths.raw_h1
        if not source.exists():
            return pd.DataFrame()
        h1 = pd.read_csv(source, parse_dates=["time"])
        return h1.sort_values("time").reset_index(drop=True)

    def _build_h1_context(self, h1: pd.DataFrame) -> pd.DataFrame:
        out = h1.copy()
        out["range"] = (out["high"] - out["low"]).astype(float)
        prev_close = out["close"].shift(1)
        tr = pd.concat(
            [
                (out["high"] - out["low"]).abs(),
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr14"] = tr.rolling(14, min_periods=1).mean()
        out["swing_high_3"] = out["high"].rolling(3, min_periods=1).max().shift(1)
        out["swing_low_3"] = out["low"].rolling(3, min_periods=1).min().shift(1)
        out["h1_bos_up"] = (out["close"] > out["swing_high_3"]).fillna(False).astype(int)
        out["h1_bos_down"] = (out["close"] < out["swing_low_3"]).fillna(False).astype(int)
        out["h1_bias"] = "NEUTRAL"
        out.loc[(out["close"] > out["close"].rolling(20, min_periods=1).mean()) | out["h1_bos_up"].eq(1), "h1_bias"] = "BULLISH"
        out.loc[(out["close"] < out["close"].rolling(20, min_periods=1).mean()) | out["h1_bos_down"].eq(1), "h1_bias"] = "BEARISH"

        candle_body = (out["close"] - out["open"]).abs()
        upper_wick = (out["high"] - out[["open", "close"]].max(axis=1)).clip(lower=0.0)
        lower_wick = (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0.0)
        avg_range = out["range"].rolling(20, min_periods=1).mean()
        out["h1_supply_rejection"] = (
            (out["h1_bias"] == "BULLISH")
            & (upper_wick > candle_body * 1.2)
            & (out["range"] >= avg_range * 1.1)
        ).astype(int)
        out["h1_demand_rejection"] = (
            (out["h1_bias"] == "BEARISH")
            & (lower_wick > candle_body * 1.2)
            & (out["range"] >= avg_range * 1.1)
        ).astype(int)

        atr_slope = out["atr14"].diff().fillna(0.0)
        compression = (out["range"] <= avg_range * 0.85).astype(int)
        weakening = (
            ((out["h1_bias"] == "BULLISH") & (out["close"].diff().rolling(3, min_periods=1).mean() <= 0))
            | ((out["h1_bias"] == "BEARISH") & (out["close"].diff().rolling(3, min_periods=1).mean() >= 0))
        ).astype(int)
        h1_exhaustion = (
            compression * 20
            + (atr_slope < 0).astype(int) * 20
            + (out["h1_supply_rejection"] | out["h1_demand_rejection"]).astype(int) * 25
            + weakening * 20
        ).clip(lower=0, upper=100)
        out["h1_exhaustion"] = h1_exhaustion.astype(float)

        out["h1_lifecycle"] = "TREND_HEALTHY"
        out.loc[compression.eq(1), "h1_lifecycle"] = "RANGE_COMPRESSION"
        out.loc[h1_exhaustion.ge(55), "h1_lifecycle"] = "TREND_EXHAUSTING"
        out.loc[h1_exhaustion.ge(75), "h1_lifecycle"] = "REVERSAL_WATCH"
        out["h1_breakout_supportive"] = (
            ((out["h1_bos_up"].eq(1)) & out["h1_bias"].eq("BULLISH"))
            | ((out["h1_bos_down"].eq(1)) & out["h1_bias"].eq("BEARISH"))
        ).astype(int)

        return out[[
            "time",
            "h1_bias",
            "h1_lifecycle",
            "h1_exhaustion",
            "h1_supply_rejection",
            "h1_demand_rejection",
            "h1_bos_up",
            "h1_bos_down",
            "h1_breakout_supportive",
        ]]

    def _build_h4_context(self, h1: pd.DataFrame) -> pd.DataFrame:
        base = h1.copy().set_index("time").sort_index()
        h4 = base.resample("4h").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        ).dropna().reset_index()
        if h4.empty:
            return pd.DataFrame(columns=["time", "h4_exhaustion", "h4_reversal_risk", "h1_bos_bias_match"])

        h4["range"] = (h4["high"] - h4["low"]).astype(float)
        prev_close = h4["close"].shift(1)
        tr = pd.concat(
            [
                (h4["high"] - h4["low"]).abs(),
                (h4["high"] - prev_close).abs(),
                (h4["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        h4["atr14"] = tr.rolling(14, min_periods=1).mean()
        h4["ma20"] = h4["close"].rolling(20, min_periods=1).mean()
        candle_body = (h4["close"] - h4["open"]).abs()
        upper_wick = (h4["high"] - h4[["open", "close"]].max(axis=1)).clip(lower=0.0)
        lower_wick = (h4[["open", "close"]].min(axis=1) - h4["low"]).clip(lower=0.0)
        avg_range = h4["range"].rolling(20, min_periods=1).mean()
        momentum = h4["close"].diff().rolling(3, min_periods=1).mean()
        h4_bias = np.where(h4["close"] >= h4["ma20"], "BULLISH", "BEARISH")
        h4_reversal_risk = (
            ((upper_wick > candle_body * 1.25) | (lower_wick > candle_body * 1.25)).astype(int) * 25
            + (h4["atr14"].diff().fillna(0.0) < 0).astype(int) * 20
            + (h4["range"] <= avg_range * 0.85).astype(int) * 20
            + (((h4_bias == "BULLISH") & (momentum <= 0)) | ((h4_bias == "BEARISH") & (momentum >= 0))).astype(int) * 25
        ).clip(lower=0, upper=100)
        h4["h4_exhaustion"] = h4_reversal_risk.astype(float)
        h4["h4_reversal_risk"] = h4_reversal_risk.astype(float)
        h4["h1_bos_bias_match"] = 0
        return h4[["time", "h4_exhaustion", "h4_reversal_risk", "h1_bos_bias_match"]]
