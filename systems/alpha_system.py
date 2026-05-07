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

        alpha_allowed = (
            (out["alpha_score"] >= 75)
            & (out.get("fake_breakout", pd.Series(0, index=out.index)) == 0)
            & (out.get("trap_probability", pd.Series(0.0, index=out.index)) < 70)
            & (out.get("multi_tf_alignment_score", pd.Series(50.0, index=out.index)) >= 65)
            & (out.get("htf_liquidity_alignment", pd.Series(0, index=out.index)) >= 0)
            & (out.get("htf_exhaustion", pd.Series(50.0, index=out.index)) < 70)
            & (~out.get("lifecycle_state", pd.Series("TREND_HEALTHY", index=out.index)).isin(["TREND_EXHAUSTING", "REVERSAL_WATCH", "FORCE_EXIT"]))
        )

        out["alpha_signal"] = np.where(alpha_allowed, "ALPHA", "")
        out["alpha_notes"] = np.where(out["alpha_signal"] == "ALPHA", "mtf_aligned_sniper", "")
        return out

    def filter_high_quality(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe[dataframe["alpha_score"] >= 75].copy()
