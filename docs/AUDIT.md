# AQRS Repository Audit

Last updated: 2026-04-29

## Current Runtime Surfaces

### V2 / Legacy Stable
- `main.py`
- `core/pipeline.py`
- `agents/`
- `strategy/pipeline_transforms.py`
- `strategy/backtesting.py`
- `strategy/replay_engine.py`
- `strategy/reporting.py`
- `strategy/*_agent.py`

### V3 / Active Production Candidate
- `main_v3.py`
- `core/v3_engine.py`
- `config/v3_config.py`
- `engines/`
- `systems/`
- `risk/risk_manager.py`
- `replay/replay_engine.py`
- `validator_institutional.py`
- `test_v3_integration.py`

### Dashboard
- `strategy/backend_api.py`
- `strategy/streamlit_app.py`
- `strategy/state_manager.py`
- `strategy/start_dashboard.py`

### MT5 Execution
- `agents/data_agent.py`
- `main_v3.py`
- `strategy/execution_agent.py`
- `strategy/execution_gate.py`
- `strategy/mt5_bridge.py`
- `strategy/verify_env.py`

## Entry Points

```powershell
python main.py --mode research
python main.py --mode backtest
python main.py --mode replay
python main_v3.py --mode research
python main_v3.py --mode backtest
python main_v3.py --mode replay
python main_v3.py --mode live
python main_v3.py --test-telegram
python -m strategy.backend_api
streamlit run strategy/streamlit_app.py
python strategy/start_dashboard.py
python test_v3_integration.py
```

## Duplicate / Overlapping Logic

- `strategy/replay_engine.py` and `replay/replay_engine.py`
- `strategy/streamlit_app.py` and `dashboard/streamlit_app.py`
- `strategy/reporting.py` and `reporting/reporting_engine.py`
- `strategy/backtesting.py` and `backtesting/backtest_engine.py`
- `strategy/execution_agent.py`, `strategy/mt5_bridge.py`, and `execution/mt5_executor.py`

## Archive Candidates

Move to `archive/` before deleting:

- `tmp_inspect_replay.py`
- `v3_final_report.py`
- `V3_PHASE1_SUMMARY.md`
- `Converter_CSV-to-SQL.py`
- `analyze_trades.py`
- `dashboard/streamlit_app.py`
- `backtesting/backtest_engine.py`
- `strategy/risk_engine.py`

## Generated Data Mixed With Source

The repository currently contains large generated CSV outputs under:

- `data/raw/`
- `data/clean/`
- `data/features/`
- `data/research/`
- `data/backtest/`
- `data/replay/`
- `data/stress/`
- `data/live/`
- `data/latest_extract/`

These should eventually be ignored or moved into an external artifact store, except for small test fixtures.

## Known Runtime Risks

- `strategy/higher_timeframe_agent.py` references `config.paths.clean_h4` and `config.paths.clean_d1`, but current `AppPaths` does not define those paths.
- V2 `core/pipeline.py` imports `strategy.execution_agent`, which now includes V3-oriented live execution behavior.
- Dashboard and validation modules expect generated CSV files to exist in specific paths.
- Some files use hardcoded fallback paths such as `Path("data/backtest")`.
- Secrets should not live in committed config. Prefer `.env` or a local ignored config file.

## Files To Avoid Touching Casually

- `main_v3.py`
- `strategy/execution_agent.py`
- `strategy/execution_gate.py`
- `strategy/mt5_bridge.py`
- `strategy/backend_api.py`
- `strategy/streamlit_app.py`
- `strategy/state_manager.py`
- `core/config.py`
- `core/v3_engine.py`
- `config/app_config.json`
- `strategy/backtesting.py`
- `strategy/replay_engine.py`
- `replay/replay_engine.py`

## Safe Refactor Sequence

1. Documentation and guardrails.
2. Archive obvious non-runtime files.
3. Add compatibility wrappers before moving imports.
4. Isolate V2 and V3 packages gradually.
5. Move generated artifacts out of tracked source paths.
6. Run compile checks and smoke tests after each step.
