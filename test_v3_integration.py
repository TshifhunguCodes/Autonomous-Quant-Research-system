#!/usr/bin/env python
"""Integration test for AQRS V3 Phase 1 completion."""

from core.config import load_config
from core.v3_engine import AQRSV3Engine
from config.v3_config import V3Config
from backtesting.backtest_engine import BacktestEngine
from research.research_engine import ResearchEngine
from reporting.reporting_engine import ReportingEngine
import pandas as pd

def main():
    base_config = load_config()
    config = V3Config.load_from(base_config)
    engine = AQRSV3Engine(config)

    # Load small dataset
    df = pd.read_csv('data/clean/xauusd_m5_clean.csv', parse_dates=['time']).head(500)
    result = engine.run_research(df=df)

    # Backtest
    backtest = BacktestEngine(config)
    metrics = backtest.compute_metrics(result)
    print('Backtest Metrics:')
    print(f'  Total Trades: {metrics["total_trades"]}')
    print(f'  Win Rate: {metrics["win_rate"]:.2%}')
    print(f'  Profit Factor: {metrics["profit_factor"]:.2f}')
    print(f'  Sharpe Ratio: {metrics["sharpe_ratio"]:.2f}')
    print(f'  Max Drawdown: {metrics["max_drawdown"]:.2%}')

    # Research
    research = ResearchEngine(config)
    artifacts = research.run(result, output_dir='data/research')
    print(f'\nResearch artifacts saved: {artifacts["meta"]}')

    # Reporting
    reporting = ReportingEngine(config)
    report = reporting.build_report(result, output_dir='data/backtest')
    print(f'\nReport generated:')
    print(f'  Alpha Trades: {report["alpha_summary"]["count"]}')
    print(f'  Flow Trades: {report["flow_summary"]["count"]}')
    print(f'  Alpha Avg Score: {report["alpha_summary"]["avg_score"]:.1f}')
    print(f'  Flow Avg Score: {report["flow_summary"]["avg_score"]:.1f}')

    print('\n✓ AQRS V3 Phase 1 Integration Test PASSED')

if __name__ == '__main__':
    main()
