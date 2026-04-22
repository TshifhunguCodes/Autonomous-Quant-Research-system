# Autonomous Quant Research System V2

This repo is a rule-based `XAUUSD` research and execution pipeline built around MetaTrader 5 data, higher-timeframe bias, M5 entries, backtesting, and guarded live execution.

## Modes

`Research`
Builds the feature and trade-setup artifacts without placing orders.

```bash
python main.py --mode research --refresh-data
```

`Backtest`
Runs the research pipeline first by default, then writes research artifacts to `data/research/` and backtest outputs to `data/backtest/`.

```bash
python main.py --mode backtest --refresh-data
python main.py --mode backtest --reuse-artifacts
```

`Live`
Runs the research pipeline, previews the latest approved trade, and only sends an order when `--execute-live` is passed and live trading is enabled in config.

```bash
python main.py --mode live --reuse-artifacts
python main.py --mode live --reuse-artifacts --execute-live
```

`Replay`
Runs a candle-by-candle historical replay that behaves like live trading but only uses past candles available at each replay step. It logs every decision, open, close, and final outcome into `data/replay/`.

```bash
python main.py --mode replay
python main.py --mode replay --replay-start 2026-04-01 --replay-end 2026-04-22
python main.py --mode replay --replay-max-candles 250
```

## What Changed In V2

- Separated `Research`, `Backtest`, and `Live` modes.
- Added JSON config management in [config/app_config.json](</c:/HTLM Clones/Autonomous-Quant-Research-system/config/app_config.json:1>).
- Added structured logging to `logs/quant_system.log`.
- Added H1 bias so M5 entries only confirm in the higher-timeframe direction.
- Made backtesting safer by writing separate outputs in `data/backtest/` instead of mutating live artifacts.
- Added live trading controls for quality threshold, H1 alignment, spread, stale signals, duplicate candles, and preview-only execution.
- V2 research artifacts now live in `data/research/`, so the old `data/features/` files remain untouched.
- Added a candle-by-candle replay engine with `replay_decisions.csv`, `replay_events.csv`, `replay_trades.csv`, and `replay_summary.csv`.

## Key Files

- [main.py](</c:/HTLM Clones/Autonomous-Quant-Research-system/main.py:1>): CLI entrypoint.
- [core/pipeline.py](</c:/HTLM Clones/Autonomous-Quant-Research-system/core/pipeline.py:1>): mode orchestration.
- [core/config.py](</c:/HTLM Clones/Autonomous-Quant-Research-system/core/config.py:1>): shared config and paths.
- [strategy/higher_timeframe_agent.py](</c:/HTLM Clones/Autonomous-Quant-Research-system/strategy/higher_timeframe_agent.py:1>): H1 context and bias.
- [strategy/backtesting.py](</c:/HTLM Clones/Autonomous-Quant-Research-system/strategy/backtesting.py:1>): realistic backtest engine.
- [strategy/execution_agent.py](</c:/HTLM Clones/Autonomous-Quant-Research-system/strategy/execution_agent.py:1>): guarded live execution.

## Notes

- `live.enabled` is `false` by default.
- Live mode still previews the latest setup even when execution is disabled.
- MT5 is only required when you refresh data or actually execute a live trade.
