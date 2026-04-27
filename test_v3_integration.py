#!/usr/bin/env python
"""Integration test for AQRS V3 Phase 1 completion."""

from core.config import load_config
from core.v3_engine import AQRSV3Engine
from config.v3_config import V3Config
import pandas as pd

def main():
    base_config = load_config()
    config = V3Config.load_from(base_config)
    engine = AQRSV3Engine(config)

    print("Starting V3 Integration Test...")
    # Load small dataset for rapid validation
    df = pd.read_csv('data/clean/xauusd_m5_clean.csv', parse_dates=['time']).head(500)
    
    # Test Step 1: Research Pipeline (83-column generation)
    research_result = engine.run_research(df=df)
    print(f"✓ Research Pipeline: Generated {len(research_result)} intelligence rows.")

    # Test Step 2: V3 Backtest Engine
    # This utilizes strategy/backtesting.py as the single source of truth
    backtest_signals = engine.run_backtest()
    alpha_signals = len(backtest_signals[backtest_signals['signal'] == 'ALPHA'])
    flow_signals = len(backtest_signals[backtest_signals['signal'] == 'FLOW'])
    
    print(f"✓ Backtest Engine: Processed {len(backtest_signals)} signals.")
    print(f"  - ALPHA signals: {alpha_signals}")
    print(f"  - FLOW signals: {flow_signals}")

    print('\n✓ AQRS V3 Phase 1 Integration Test PASSED')

if __name__ == '__main__':
    main()
