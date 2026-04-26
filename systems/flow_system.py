from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class FlowSystem:
    """Adaptive exploratory engine for broader FLOW setups."""

    def __init__(self, config: Any):
        self.config = config

    def generate_flow_setups(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["flow_score"] = 20
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

        out["flow_signal"] = np.where(out["flow_score"] >= 55, "FLOW", "")
        out["flow_notes"] = np.where(out["flow_signal"] == "FLOW", "exploratory", "")
        return out

    def score_experimental_patterns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        out = dataframe.copy()
        out["experiment_score"] = 0
        out.loc[out["pattern"] == "CHOCH", "experiment_score"] += 15
        out.loc[out["pattern"].isin(["DOUBLE_TOP", "DOUBLE_BOTTOM"]), "experiment_score"] += 10
        out.loc[out["fvg_zone"] == 1, "experiment_score"] += 8
        return out
