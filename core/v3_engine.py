from __future__ import annotations
from typing import Any

import pandas as pd

from config.v3_config import V3Config
from engines.behavior_engine import MarketBehaviorEngine
from engines.structure_engine import PriceActionStructureEngine
from engines.zone_engine import ZoneEngine
from replay.replay_engine import ReplayEngine
from risk.risk_manager import RiskManager
from systems.alpha_system import AlphaSystem
from systems.flow_system import FlowSystem


class AQRSV3Engine:
    """AQRS V3 orchestrator for research, replay, and live execution."""

    def __init__(self, config: V3Config):
        self.config = config
        self.behavior = MarketBehaviorEngine(config)
        self.structure = PriceActionStructureEngine(config)
        self.zone = ZoneEngine(config)
        self.alpha = AlphaSystem(config)
        self.flow = FlowSystem(config)
        self.risk = RiskManager(config)
        self.replay = ReplayEngine(config)

    def _load_data(self, custom_path: str | None = None) -> pd.DataFrame:
        """Loads historical data, defaulting to config paths if no custom path is provided."""
        source = custom_path or self.config.base.paths.clean_m5
        if not source.exists():
            source = self.config.base.paths.raw_m5
        if not source.exists():
            raise FileNotFoundError(f"V3 Engine Error: Could not locate historical data at {source}")
        df = pd.read_csv(source, parse_dates=["time"])
        return df

    def run_research(self, df: pd.DataFrame | None = None, refresh_data: bool = False) -> pd.DataFrame:
        if df is None:
            df = self._load_data()
        pipeline = self.behavior.classify_market(df)
        pipeline = self.structure.build_price_action_structure(pipeline)
        pipeline = self.zone.build_zones(pipeline)
        pipeline = self.alpha.generate_alpha_setups(pipeline)
        pipeline = self.flow.generate_flow_setups(pipeline)
        pipeline = self._resolve_signals(pipeline)
        pipeline = self.risk.annotate_trade_risk(pipeline)
        return pipeline

    def run_backtest(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.run_research(df=df)

    def run_replay(
        self,
        df: pd.DataFrame | None = None,
        start: str | None = None,
        end: str | None = None,
        max_candles: int | None = None,
    ) -> pd.DataFrame:
        if df is None:
            df = self._load_data()
        return self.replay.run(df=df, start=start, end=end, max_candles=max_candles)

    def _resolve_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = "NO_TRADE"
        out.loc[out["alpha_signal"] == "ALPHA", "signal"] = "ALPHA"
        flow_mask = (out["signal"] == "NO_TRADE") & (out["flow_signal"] == "FLOW")
        out.loc[flow_mask, "signal"] = "FLOW"
        out["signal_owner"] = out["signal"]

        # Backwards compatibility: unified score for dashboard and validator
        out["confirm_score"] = 0.0
        out.loc[out["signal"] == "ALPHA", "confirm_score"] = out["alpha_score"]
        out.loc[out["signal"] == "FLOW", "confirm_score"] = out["flow_score"]

        return out
