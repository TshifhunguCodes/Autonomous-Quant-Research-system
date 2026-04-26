from __future__ import annotations
from typing import Any
from pathlib import Path

import pandas as pd


class ResearchEngine:
    """Research engine to generate features, structure, regimes, signals, and setups."""

    def __init__(self, config: Any):
        self.config = config

    def run(self, df: pd.DataFrame, output_dir: str | None = None) -> dict[str, Any]:
        """Run end-to-end research mode and produce artifact outputs."""
        artifacts = {
            "pipeline": df,
            "meta": {
                "rows": len(df),
                "alpha_signals": len(df[df["signal"] == "ALPHA"]),
                "flow_signals": len(df[df["signal"] == "FLOW"]),
            },
        }

        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)

            df.to_csv(path / "xauusd_m5_features.csv", index=False)

            df[df["signal"] != "NO_TRADE"].to_csv(path / "signals.csv", index=False)
            df[df["signal"] == "ALPHA"].to_csv(path / "alpha_setups.csv", index=False)
            df[df["signal"] == "FLOW"].to_csv(path / "flow_setups.csv", index=False)

            structure_cols = [c for c in df.columns if "structure" in c or "pattern" in c or "bos" in c]
            if structure_cols:
                df[["time"] + structure_cols].to_csv(path / "structure.csv", index=False)

            zone_cols = [c for c in df.columns if "zone" in c or "level" in c or "order_block" in c or "fvg" in c]
            if zone_cols:
                df[["time"] + zone_cols].to_csv(path / "zones.csv", index=False)

            behavior_cols = [c for c in df.columns if "behavior" in c or "momentum" in c or "trend" in c]
            if behavior_cols:
                df[["time"] + behavior_cols].to_csv(path / "market_state.csv", index=False)

        return artifacts
