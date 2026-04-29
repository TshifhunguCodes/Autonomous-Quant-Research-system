#!/usr/bin/env python
"""Integration test for AQRS V3 Phase 1 completion."""

import pandas as pd

from config.v3_config import V3Config
from core.config import load_config
from core.v3_engine import AQRSV3Engine


def main():
    base_config = load_config()
    config = V3Config.load_from(base_config)
    engine = AQRSV3Engine(config)

    print("Starting V3 Integration Test...")
    df = pd.read_csv("data/clean/xauusd_m5_clean.csv", parse_dates=["time"]).head(500)

    research_result = engine.run_research(df=df)
    required_columns = {"signal", "confirmed_signal", "quality", "entry_price", "stop_loss", "take_profit"}
    missing_columns = required_columns - set(research_result.columns)
    if missing_columns:
        raise AssertionError(f"Research output is missing required columns: {sorted(missing_columns)}")
    print(f"[OK] Research Pipeline: Generated {len(research_result)} intelligence rows.")

    backtest_signals = engine.run_backtest()
    alpha_signals = len(backtest_signals[backtest_signals["signal"] == "ALPHA"])
    flow_signals = len(backtest_signals[backtest_signals["signal"] == "FLOW"])

    print(f"[OK] Backtest Engine: Processed {len(backtest_signals)} signals.")
    print(f"  - ALPHA signals: {alpha_signals}")
    print(f"  - FLOW signals: {flow_signals}")

    replay_result = engine.run_replay(df=df, max_candles=50)
    if replay_result.empty:
        raise AssertionError("Replay output is empty.")
    print(f"[OK] Replay Engine: Generated {len(replay_result)} replay rows.")

    print("\n[OK] AQRS V3 Phase 1 Integration Test PASSED")


if __name__ == "__main__":
    main()
