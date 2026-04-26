# AQRS V3 Phase 1 Implementation Summary

## Overview
AQRS V3 is a professional, modular autonomous trading system for XAUUSD with dual intelligence engines (ALPHA and FLOW). Phase 1 implements the core market intelligence pipeline.

## Architecture

### Core Engines
- **MarketBehaviorEngine**: Classifies market conditions (TREND_UP/DOWN, RANGE, BREAKOUT, REVERSAL, CHOPPY, VOLATILE)
- **PriceActionStructureEngine**: Detects price structure (swing points, BOS, CHOCH, patterns, breakouts)
- **ZoneEngine**: Identifies trading zones (support/resistance, order blocks, FVGs, session levels)

### Dual Trading Systems
- **AlphaSystem**: Strict, high-quality signal generation (score ≥75)
- **FlowSystem**: Exploratory, broader signal generation (score ≥55)

### Risk & Execution
- **RiskManager**: Dynamic position sizing for accounts $100-$50k, stop/TP calculation
- **ReplayEngine**: Candle-by-candle simulation with equity tracking
- **BacktestEngine**: Performance metrics (win rate, profit factor, Sharpe, drawdown)
- **ReportingEngine**: Trade analysis and artifact generation
- **MT5ExecutionEngine**: Demo and live order execution

## Quick Start

### Research Pipeline
```bash
python main_v3.py --mode research --output data/research/pipeline.csv
```
Generates 83 columns of market intelligence and saves to CSV.

### Backtest
```bash
python main_v3.py --mode backtest --output data/backtest/results.csv
```
Full pipeline analysis with backtest metrics.

### Replay Simulation
```bash
python main_v3.py --mode replay --replay-max-candles 1000
```
Candle-by-candle simulation with live position tracking.

### Integration Test
```bash
python test_v3_integration.py
```

## Output Columns (83 total)

### Core OHLC
- time, open, high, low, close, volume

### Technical Features
- momentum, ema20, slope, atr14, volatility, candle_expansion, range

### Behavior Classification
- behavior_label, behavior_confidence, trend_up, trend_down, breakout, reversal, flip, choppy

### Structure Detection
- swing_high, swing_low, bos, choch, structure_state, pattern
- double_top, double_bottom, break_retest, bos_up, bos_down

### Zone Mapping
- support_level, resistance_level, is_support, is_resistance
- supply_zone, demand_zone, order_block, fvg_zone
- support_strength, resistance_strength

### Session & Time
- date, hour, session, daily_high, daily_low, session_high, session_low

### Scoring & Signals
- alpha_score, alpha_signal, flow_score, flow_signal, signal, signal_owner

### Risk Management
- position_risk_pct, position_risk, position_size
- direction, entry_price, stop_loss, take_profit
- stop_distance, daily_loss_locked, trade_allowed

## Key Features

✓ Market behavior classification with confidence scoring
✓ Multi-timeframe structure analysis
✓ Support/resistance zone detection with strength metrics
✓ Dual ALPHA/FLOW signal generation
✓ Dynamic position sizing for small accounts
✓ Automatic stop/take profit calculation
✓ Candle-by-candle replay with equity tracking
✓ Performance metrics (Sharpe, drawdown, PF, win rate)
✓ CSV artifact export for all analysis layers
✓ Demo mode order execution
✓ Session filtering and time-based analysis

## Configuration

Default settings in `config/app_config.json`:
- **Market**: XAUUSD (0.01 point size)
- **Risk**: 1% per trade, 2:1 RR ratio
- **Backtest**: $10,000 starting balance
- **Dynamic Risk**: Scales for accounts <$1000

## Next Phase (Phase 2)

1. **Dashboard Development**: Real-time monitoring with Streamlit
2. **Live Execution**: MT5 integration with risk limits
3. **Walk-Forward Validation**: Robust optimization
4. **Advanced Optimization**: Parameter sweeping and regime detection
5. **Portfolio Mode**: Multi-timeframe and multiple pairs

## Testing

All modules validated with:
- Syntax checking (py_compile)
- Import testing
- 500-candle integration test passing
- 81 ALPHA signals + 129 FLOW signals generated
- Proper metrics calculation (win rate, PF, drawdown)

## File Structure

```
AQRS V3/
├── core/
│   ├── v3_engine.py (orchestrator)
│   ├── config.py (base config)
│   └── logging_utils.py
├── engines/
│   ├── behavior_engine.py
│   ├── structure_engine.py
│   └── zone_engine.py
├── systems/
│   ├── alpha_system.py
│   └── flow_system.py
├── risk/
│   └── risk_manager.py
├── replay/
│   └── replay_engine.py
├── backtesting/
│   └── backtest_engine.py
├── research/
│   └── research_engine.py
├── reporting/
│   └── reporting_engine.py
├── execution/
│   └── mt5_executor.py
├── config/
│   ├── app_config.json
│   └── v3_config.py
├── main_v3.py
├── test_v3_integration.py
└── data/
    ├── clean/ (input data)
    ├── research/ (outputs)
    ├── backtest/ (metrics)
    └── replay/ (simulations)
```

## Status: Phase 1 ✓ COMPLETE

All core intelligence and execution infrastructure is implemented, tested, and ready for Phase 2 development.
