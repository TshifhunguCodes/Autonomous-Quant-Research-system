from __future__ import annotations
from typing import Any
from pathlib import Path

import pandas as pd


class ReportingEngine:
    """Reporting engine for performance dashboards and walk-forward summaries."""

    def __init__(self, config: Any):
        self.config = config

    def build_report(self, df: pd.DataFrame, output_dir: str | None = None) -> dict[str, Any]:
        """Generate backtest and session reports."""
        output = {}

        alpha_trades = df[df["signal"] == "ALPHA"].copy()
        flow_trades = df[df["signal"] == "FLOW"].copy()

        output["alpha_summary"] = {
            "count": len(alpha_trades),
            "avg_score": float(alpha_trades["alpha_score"].mean()) if len(alpha_trades) > 0 else 0.0,
            "min_score": float(alpha_trades["alpha_score"].min()) if len(alpha_trades) > 0 else 0.0,
            "max_score": float(alpha_trades["alpha_score"].max()) if len(alpha_trades) > 0 else 0.0,
        }

        output["flow_summary"] = {
            "count": len(flow_trades),
            "avg_score": float(flow_trades["flow_score"].mean()) if len(flow_trades) > 0 else 0.0,
            "min_score": float(flow_trades["flow_score"].min()) if len(flow_trades) > 0 else 0.0,
            "max_score": float(flow_trades["flow_score"].max()) if len(flow_trades) > 0 else 0.0,
        }

        output["session_summary"] = self._session_breakdown(df)
        output["behavior_summary"] = self._behavior_breakdown(df)

        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)

            alpha_trades.to_csv(path / "alpha_trades.csv", index=False)
            flow_trades.to_csv(path / "flow_trades.csv", index=False)

            summary_df = pd.DataFrame([
                {
                    "type": "ALPHA",
                    "count": output["alpha_summary"]["count"],
                    "avg_score": output["alpha_summary"]["avg_score"],
                },
                {
                    "type": "FLOW",
                    "count": output["flow_summary"]["count"],
                    "avg_score": output["flow_summary"]["avg_score"],
                },
            ])
            summary_df.to_csv(path / "signal_summary.csv", index=False)

        return output

    def _session_breakdown(self, df: pd.DataFrame) -> dict[str, Any]:
        sessions = {}
        for session in ["ASIA", "LONDON", "NEW_YORK"]:
            session_data = df[df["session"] == session]
            sessions[session] = {
                "total_bars": len(session_data),
                "signal_count": len(session_data[session_data["signal"] != "NO_TRADE"]),
                "avg_alpha_score": float(session_data["alpha_score"].mean()) if len(session_data) > 0 else 0.0,
                "avg_flow_score": float(session_data["flow_score"].mean()) if len(session_data) > 0 else 0.0,
            }
        return sessions

    def _behavior_breakdown(self, df: pd.DataFrame) -> dict[str, Any]:
        behaviors = {}
        for behavior in df["behavior_label"].unique():
            behavior_data = df[df["behavior_label"] == behavior]
            behaviors[behavior] = {
                "count": len(behavior_data),
                "signal_count": len(behavior_data[behavior_data["signal"] != "NO_TRADE"]),
                "avg_alpha_score": float(behavior_data["alpha_score"].mean()),
                "avg_flow_score": float(behavior_data["flow_score"].mean()),
            }
        return behaviors
