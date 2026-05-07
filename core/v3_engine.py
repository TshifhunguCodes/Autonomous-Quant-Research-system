from __future__ import annotations
from typing import Any

import pandas as pd

from agents.cleaning_agent import run as clean_data
from agents.data_agent import run as fetch_data
from config.v3_config import V3Config
from engines import MarketBehaviorEngine, PriceActionStructureEngine, ZoneEngine
from engines.liquidity_engine import LiquidityEngine
from engines.market_lifecycle_engine import MarketLifecycleEngine
from engines.mtf_context_engine import MTFContextEngine
from replay.replay_engine import ReplayEngine
from risk.risk_manager import RiskManager
from systems import AlphaSystem, FlowSystem


class AQRSV3Engine:
    """AQRS V3 orchestrator for research, replay, and live execution."""

    def __init__(self, config: V3Config):
        self.config = config
        self.behavior = MarketBehaviorEngine(config)
        self.structure = PriceActionStructureEngine(config)
        self.zone = ZoneEngine(config)
        self.lifecycle = MarketLifecycleEngine(config)
        self.liquidity = LiquidityEngine(config)
        self.mtf_context = MTFContextEngine(config)
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
        if refresh_data:
            fetch_data(self.config.base)
            clean_data(self.config.base)
        if df is None:
            df = self._load_data()
        pipeline = self.behavior.classify_market(df)
        pipeline = self.structure.build_price_action_structure(pipeline)
        pipeline = self.zone.build_zones(pipeline)
        pipeline = self.lifecycle.classify_lifecycle(pipeline)
        pipeline = self.liquidity.classify_liquidity(pipeline)
        pipeline = self.mtf_context.classify_context(pipeline)
        pipeline = self.alpha.generate_alpha_setups(pipeline)
        pipeline = self.flow.generate_flow_setups(pipeline)
        pipeline = self._resolve_signals(pipeline)
        pipeline = self._apply_mtf_confidence(pipeline)
        pipeline = self.risk.annotate_trade_risk(pipeline)
        pipeline = self._annotate_execution_compatibility(pipeline)
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

    def _apply_mtf_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        aligned = out.get("multi_tf_alignment_score", pd.Series(50.0, index=out.index)).fillna(50.0)
        boost = pd.Series(0.0, index=out.index)
        boost.loc[aligned >= 80] = 10.0
        boost.loc[(aligned >= 65) & (aligned < 80)] = 5.0
        out["confirm_score"] = (out["confirm_score"] + boost).clip(lower=0.0, upper=100.0)
        return out

    def _annotate_execution_compatibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add legacy execution columns expected by the live execution agent."""
        out = df.copy()
        out["quality"] = "NONE"
        out.loc[out["confirm_score"] >= 55, "quality"] = "MEDIUM"
        out.loc[out["confirm_score"] >= 70, "quality"] = "HIGH"
        out.loc[out["confirm_score"] >= 85, "quality"] = "ELITE"
        out.loc[out["signal"] == "NO_TRADE", "quality"] = "NONE"
        out["confirmed_signal"] = "no_trade"
        out.loc[(out["signal"].isin(["ALPHA", "FLOW"])) & (out["direction"] == "LONG"), "confirmed_signal"] = "buy"
        out.loc[(out["signal"].isin(["ALPHA", "FLOW"])) & (out["direction"] == "SHORT"), "confirmed_signal"] = "sell"
        out["market_regime"] = out["behavior_label"]
        out["market_state"] = out["behavior_label"]

        return out
