# Autonomous Quant Research System

AQRS is a Python trading research and execution system for `XAUUSD` built around MetaTrader 5 data, M5 market structure, replay/backtesting, Streamlit monitoring, and guarded demo execution.

The repository currently contains two generations:

- **V2 legacy pipeline**: stable rule-based research/backtest/replay workflow via `main.py`.
- **V3 active model**: modular behavior, structure, zone, Alpha/Flow, risk, replay, validation, Telegram alerting, and MT5 demo execution via `main_v3.py`.

## Current Status

V3 is the active development and live-demo path. V2 remains available while the repo is being safely reorganized.

Before large refactors, read:

- `docs/AUDIT.md`
- `project_structure/System_Documentation.md`

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

MT5 requirements:

- MetaTrader 5 desktop terminal installed.
- Logged into a demo account.
- XAUUSD visible/selected in Market Watch.
- Algo trading enabled if using `--execute`.

## Secrets

Preferred local setup:

```powershell
$env:TELEGRAM_BOT_TOKEN="your_botfather_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
```

You can also copy `.env.example` to `.env` for your own local notes, but the current app reads environment variables and `config/app_config.json`.

Do not commit real account credentials or live broker secrets.

## Telegram Test

```powershell
python main_v3.py --test-telegram
```

Expected success:

```text
Telegram connected.
Bot: @your_bot_name
Chat ID: your_chat_id
Test message id: 123
```

## V3 Commands

Research:

```powershell
python main_v3.py --mode research --output data/research/pipeline.csv
```

Backtest and readiness validation:

```powershell
python main_v3.py --mode backtest --output data/backtest/v3_research_output.csv
```

Backtest without readiness validation:

```powershell
python main_v3.py --mode backtest --output data/backtest/v3_research_output.csv --skip-readiness
```

Replay:

```powershell
python main_v3.py --mode replay --replay-max-candles 1000 --output data/replay/replay_decisions.csv
```

Live preview, no orders:

```powershell
python main_v3.py --mode live --run-days 20 --poll-seconds 30 --live-lookback-days 7 --relaxed-demo-gate
```

Live demo execution:

```powershell
python main_v3.py --mode live --execute --run-days 20 --poll-seconds 30 --live-lookback-days 7 --relaxed-demo-gate
```

Notes:

- The live runner must keep the PC awake, MT5 open, and internet connected.
- `strategy/mt5_bridge.py` currently blocks non-demo accounts.
- Telegram trade alerts are sent before execution.
- `--relaxed-demo-gate` lets V3 Alpha/Flow signals pass without strict SMC/session/spread-style blockers for observation.

## Dashboard

Start backend and Streamlit together:

```powershell
python strategy/start_dashboard.py
```

Or manually:

```powershell
python -m strategy.backend_api
streamlit run strategy/streamlit_app.py
```

Backend default:

```text
http://127.0.0.1:8001
```

## V2 Commands

Research:

```powershell
python main.py --mode research --refresh-data
```

Backtest:

```powershell
python main.py --mode backtest --refresh-data
python main.py --mode backtest --reuse-artifacts
```

Replay:

```powershell
python main.py --mode replay
python main.py --mode replay --replay-start 2026-04-01 --replay-end 2026-04-22
python main.py --mode replay --replay-max-candles 250
```

Live preview/execution:

```powershell
python main.py --mode live --reuse-artifacts
python main.py --mode live --reuse-artifacts --execute-live
```

## Important Files

- `main.py`: V2 CLI.
- `main_v3.py`: V3 CLI, live runner, Telegram test.
- `core/config.py`: shared config and path registry.
- `config/app_config.json`: main runtime config.
- `core/pipeline.py`: V2 orchestration.
- `core/v3_engine.py`: V3 orchestration.
- `engines/`: V3 market behavior, structure, and zone engines.
- `systems/`: V3 Alpha and Flow systems.
- `risk/risk_manager.py`: V3 risk annotation.
- `replay/replay_engine.py`: V3 replay.
- `strategy/backend_api.py`: dashboard API.
- `strategy/streamlit_app.py`: main dashboard UI.
- `strategy/mt5_bridge.py`: MT5 demo execution guard.
- `strategy/execution_agent.py`: live signal execution.

## Generated Outputs

Generated CSVs are written under `data/`, especially:

- `data/raw/`
- `data/clean/`
- `data/research/`
- `data/backtest/`
- `data/replay/`
- `data/live/`
- `data/stress/`
- `data/latest_extract/`

Long-term, generated artifacts should be kept out of source control or moved to a dedicated artifact store.

## Safety Checklist

Before live demo execution:

1. Confirm MT5 is on a demo account.
2. Run `python main_v3.py --test-telegram`.
3. Run live preview first without `--execute`.
4. Confirm `data/research/trade_setups.csv` updates.
5. Then run with `--execute` if ready.

## Refactor Policy

This repo is being modernized step by step. Do not move production files without adding compatibility wrappers and running checks.

Required checks after code changes:

```powershell
python -m compileall -q .
python test_v3_integration.py
python main_v3.py --mode backtest --skip-readiness
```
