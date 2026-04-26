from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class ZoneEngine:
    """Automatically generates zones and scores their strength and freshness."""

    def __init__(self, config: Any):
        self.config = config

    def build_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["time"] = pd.to_datetime(out["time"])
        out = out.sort_values("time").reset_index(drop=True)

        out["date"] = out["time"].dt.date
        out["hour"] = out["time"].dt.hour
        out["session"] = out["hour"].apply(self._session_label)

        daily = out.groupby("date").agg(
            daily_high=("high", "max"),
            daily_low=("low", "min"),
        )
        out = out.merge(daily, on="date", how="left")
        out["session_high"] = (out["high"] == out["daily_high"]).astype(int)
        out["session_low"] = (out["low"] == out["daily_low"]).astype(int)

        out["support_level"] = out["low"].rolling(30, min_periods=1).min()
        out["resistance_level"] = out["high"].rolling(30, min_periods=1).max()
        out["is_support"] = (out["close"] <= out["support_level"] * 1.01).astype(int)
        out["is_resistance"] = (out["close"] >= out["resistance_level"] * 0.99).astype(int)

        out["supply_zone"] = 0
        out["demand_zone"] = 0
        out.loc[out["high"] >= out["resistance_level"].shift(1), "supply_zone"] = 1
        out.loc[out["low"] <= out["support_level"].shift(1), "demand_zone"] = 1

        out["order_block"] = 0
        previous_bear = out["close"].shift(1) < out["open"].shift(1)
        strong_rally = out["close"] > out["open"]
        out.loc[previous_bear & strong_rally, "order_block"] = 1

        out["fvg_zone"] = 0
        gap_up = (out["open"] > out["high"].shift(1)) & (out["low"] > out["high"].shift(1))
        gap_down = (out["open"] < out["low"].shift(1)) & (out["high"] < out["low"].shift(1))
        out.loc[gap_up | gap_down, "fvg_zone"] = 1

        out["support_strength"] = (
            out["is_support"] * 1.0
            + out["session_low"] * 0.5
            + out["demand_zone"] * 0.8
        )
        out["resistance_strength"] = (
            out["is_resistance"] * 1.0
            + out["session_high"] * 0.5
            + out["supply_zone"] * 0.8
        )
        return out

    def annotate_zone_strength(self, zone: dict[str, Any]) -> dict[str, Any]:
        zone["strength"] = float(zone.get("strength", 0.0))
        zone["freshness"] = float(zone.get("freshness", 1.0))
        zone["break_probability"] = float(zone.get("break_probability", 0.5))
        zone["bounce_probability"] = float(zone.get("bounce_probability", 0.5))
        return zone

    def _session_label(self, hour: int) -> str:
        if 0 <= hour < 8:
            return "ASIA"
        if 8 <= hour < 16:
            return "LONDON"
        return "NEW_YORK"
