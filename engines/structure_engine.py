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
