from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class AlphaSystem:
    """Precision sniper engine for high-quality ALPHA setups."""

    def __init__(self, config: Any):
        self.config = config

    def generate_alpha_setups(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["alpha_score"] = 0
        out.loc[out["behavior_label"].isin(["TREND_UP", "TREND_DOWN"]), "alpha_score"] += 20
        out.loc[out["structure_state"].isin(["HH", "LL"]), "alpha_score"] += 15
        out.loc[out["bos"] == 1, "alpha_score"] += 15
        out.loc[out["break_retest"] == 1, "alpha_score"] += 12
        out.loc[out["pattern"].isin(["DOUBLE_TOP", "DOUBLE_BOTTOM"]), "alpha_score"] += 10
        out.loc[out["order_block"] == 1, "alpha_score"] += 10
        out.loc[out["is_support"] == 1, "alpha_score"] += 10
        out.loc[out["is_resistance"] == 1, "alpha_score"] += 10
        out.loc[out["session"].isin(["LONDON", "NEW_YORK"]), "alpha_score"] += 8
        out.loc[out["behavior_confidence"] >= 70, "alpha_score"] += 8
        out.loc[out["volatility"] == 1, "alpha_score"] -= 5
        out.loc[out["choppy"] == 1, "alpha_score"] -= 10

        out["alpha_signal"] = np.where(out["alpha_score"] >= 75, "ALPHA", "")
        out["alpha_notes"] = np.where(out["alpha_signal"] == "ALPHA", "strict_alpha", "")
        return out

    def filter_high_quality(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe[dataframe["alpha_score"] >= 75].copy()
