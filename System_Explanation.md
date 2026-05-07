# AQRS V3: System File Explanation

This document provides a detailed breakdown of every file within the Autonomous Quant Research System (AQRS). It explains their roles, core logic, and inter-dependencies within the V3 institutional trading pipeline.

## 📂 Root Directory
*   **`main.py`**: The entry point for the **V2 Legacy Pipeline**. It manages research, backtesting, and replay modes for the older version of the system.
*   **`main_v3.py`**: The primary entry point for the **V3 Production System**. Now optimized for **Persistent MT5 Connectivity** to minimize signal latency. It orchestrates the 83-column pipeline and hands live signals directly to the Execution Agent.
*   **`README.md`**: The general setup guide, secret management instructions, and quick-start command list.
*   **`Converter_CSV-to-SQL.py`**: A utility script used to export backtest results from CSV files into a PostgreSQL database for advanced data analysis.
*   **`validator_institutional.py`**: The final gatekeeper. It runs a 20-point diagnostic on backtest results to calculate the "Readiness Score" before going live.
*   **`test_v3_integration.py`**: A validation script that runs a localized test of the V3 engine to ensure all modules are communicating correctly.
*   **`analyze_trades.py`**: (Utility) Parses MT5 history to provide a deep dive into execution quality, slippage, and spread impact.
*   **`v3_final_report.py`**: (Utility) Consolidates results from multiple backtest runs into a single executive summary.
*   **`V3_PHASE1_SUMMARY.md`**: A technical summary of the Phase 1 implementation, defining the core engines and the complete 83-column intelligence schema.
*   **`System_Explanation.md`**: (This file) A comprehensive directory and file map for system maintenance and AI indexing.

## 📂 `core/` (The Backbone)
*   **`config.py`**: Defines the `AppConfig` and `AppPaths` DataClasses. It is responsible for safe path resolution and loading system-wide settings.
*   **`logging_utils.py`**: Provides a standardized logging configuration so that all modules output consistent, time-stamped logs to both the console and log files.
*   **`pipeline.py`**: Contains the orchestration logic for the V2 system.
*   **`v3_engine.py`**: The "Master Orchestrator" for V3. It integrates the modular Market Behavior, Structure, and Zone engines into a single research pipeline.

## 📂 `research/` (Pipeline Generation)
*   **`research_engine.py`**: The V3 research engine that runs the full 83-column market intelligence pipeline and generates the `pipeline.csv` artifact.

## 📂 `backtesting/` (Historical Performance)
*   **`backtest_engine.py`**: A dedicated engine for running multi-system backtests and generating performance metrics (Profit Factor, Sharpe).

## 📂 `engines/` (The Intelligence Core)
*   **`behavior_engine.py`**: Specialized logic for labeling the current market environment (e.g., TREND_UP vs. CHOPPY). It provides the confidence levels for regime shifts.
*   **`structure_engine.py`**: Dedicated price action engine. It tracks the lifecycle of swing points and identifies Break of Structure (BOS) or Change of Character (CHOCH).
*   **`zone_engine.py`**: The mapping tool. It identifies institutional Supply/Demand zones and calculates "Zone Strength" based on how many times a level has been tested.

## 📂 `systems/` (The Strategy Logic)
*   **`alpha_system.py`**: Contains the entry/exit logic for the **ALPHA (Sniper)** system. It focuses on high-conviction, SMC-aligned setups with strict session filters.
*   **`flow_system.py`**: Contains the logic for the **FLOW (Sensor)** system. It is designed to be highly active, providing the data needed for adaptive learning.

## 📂 `risk/` (Capital Preservation)
*   **`risk_manager.py`**: The brain for position sizing. It uses account equity and current ATR to determine the exact lot size to risk exactly 1% (or your configured amount) per trade.

## 📂 `replay/` (Forensic Simulation)
*   **`replay_engine.py`**: (V3 Implementation) A more advanced version of the legacy replay engine that handles asynchronous artifact generation and detailed equity tracking for V3 signals.

## 📂 `reporting/` (Performance Analytics)
*   **`reporting_engine.py`**: The module responsible for generating Profit Factor, Win Rate, and Drawdown metrics. It creates the visual artifacts used by the dashboard.

## 📂 `execution/` (The Sniper's Trigger)
*   **`mt5_executor.py`**: A low-level wrapper for `MetaTrader5` commands. It specializes in order routing and handling broker-specific error codes (e.g., 10016).

## 📂 `data/` (The Persistence Layer)
*   **`raw/`**: Contains raw, unmodified market data exported from MetaTrader 5 (e.g., `xauusd_m5.csv`).
*   **`clean/`**: Contains cleaned and time-aligned market data ready for feature engineering (e.g., `xauusd_m5_clean.csv`).
*   **`research/`**: Stores the output of the research pipeline, including `pipeline.csv` (83 intelligence columns) and `trade_setups.csv`.
*   **`backtest/`**: Stores backtesting artifacts like `results.csv`, trade lists, and performance summaries.
*   **`replay/`**: Stores data related to forensic replay simulations, such as `replay_decisions.csv` and `replay_trades.csv`.
*   **`live/`**: Contains live execution logs, including `execution_audit.csv` and `trade_outcomes.csv` used for adaptive learning.
*   **`stress/`**: Stores results from random walk-forward stress tests.
*   **`latest_extract/`**: Temporary storage for the most recent data extracts from MT5.

## 📂 `strategy/` (The Brains & Execution)
### 🛠️ Strategic Engines
*   **`pipeline_transforms.py`**: The primary data transformation hub. It contains the logic for feature engineering (ATR, RSI, moving averages), market structure detection (HH/LL/LH/HL), support/resistance zone identification, and signal confirmation scoring.
*   **`smc_ict_engine.py`**: An advanced overlay that detects "Smart Money Concepts" like Order Blocks, Fair Value Gaps (FVG), and Market Structure Shifts (MSS).
*   **`volume_profile_engine.py`**: Identifies institutional levels by calculating the Point of Control (POC) and Value Areas based on tick volume.
*   **`signal_validation_engine.py`**: The "Math Firewall." It enforces institutional rules such as TP/SL integrity, minimum risk-reward ratios (default 1.5), and signal deduplication before execution.

### 🚀 Execution & MT5 Connectivity
*   **`execution_agent.py`**: Orchestrates the live order process. It performs final price normalization, sends Telegram trade alerts, and prepares the MT5 order request with protective stops.
*   **`execution_gate.py`**: The strategic safety filter. It applies institutional filters like ICT Kill Zones, spread-to-ATR thresholds, slippage guards, and adaptive learning risk multipliers.
*   **`mt5_bridge.py`**: The primary gateway to MT5. It enforces demo-only trading, checks daily drawdown limits (e.g., 2%), manages max simultaneous positions, and syncs trade outcomes.

### 📊 Backtesting & Replay
*   **`backtesting.py`**: A vectorized simulation engine for System A (Alpha) and System B (Flow). It calculates equity curves and manages break-even stop logic.
*   **`replay_engine.py`**: A forensic simulator that processes the market candle-by-candle, allowing you to see exactly how the system would have reacted in real-time.
*   **`reporting.py`**: Generates detailed performance artifacts, including monthly summaries, equity curves, and rolling stability reports.
*   **`stress_testing.py`**: Performs random "Walk-Forward" simulations to ensure the strategy isn't just lucky but is actually robust across different market slices.

### 📈 Dashboard & API
*   **`backend_api.py`**: A FastAPI server that provides a real-time data bridge between the MT5 platform, local CSV files, and the web dashboard.
*   **`streamlit_app.py`**: The frontend code for the web-based dashboard. It provides the visual interface for monitoring trades, charts, and system intelligence.
*   **`state_manager.py`**: Coordinates the data flow for the dashboard, ensuring that Replay and Live data are parsed correctly for the UI.
*   **`start_dashboard.py`**: A convenience script that launches both the Backend API and the Streamlit UI simultaneously.

### 🤖 Pipeline Agents (Modular Wrappers)
*   **`market_state_agent.py`**: Identifies Trend vs. Range vs. Choppy states using volatility and flip counts.
*   **`structure_agent.py`**: Tracks HH/LL transitions and identifies market structure phases.
*   **`regime_agent.py`**: Maps market states to risk multipliers (e.g., Aligned Trend = 1.0x).
*   **`signal_agent.py`**: Generates raw bias (Buy/Sell) based on price action patterns.
*   **`setup_engine.py`**: Filters raw signals into specific setups like Retests or Breakouts.
*   **`confirmation_agent.py`**: Scores setups using candle wicks, volume, and momentum alignment.
*   **`entry_agent.py`**: Finalizes entry price, stop-loss, and take-profit calculations using ATR offsets.
*   **`higher_timeframe_agent.py`**: Analyzes H1/H4 data to provide institutional trend alignment.
*   **`risk_engine.py`**: (Legacy) Older version of the risk manager, primarily used for V2 backtesting compatibility.

### 🛡️ Maintenance & Utility
*   **`verify_env.py`**: A diagnostic tool that checks if Python dependencies and MT5 connections are correctly configured.
*   **`smoke_test.py`**: A quick reliability test to verify that the core research transforms aren't broken after a code change.
*   **`debug_engine.py`**: Outputs a terminal-based "Quality Audit" of the current pipeline results.
*   **`decision_report.py`**: Generates governance reports that advise which market sessions are currently safe to trade.
*   **`README.md`**: Maintenance notes for operational tools.

## 📂 `config/` (The Settings)
*   **`app_config.json`**: The main configuration file containing symbol settings, risk parameters, and lot sizes.
*   **`v3_config.py`**: Python-based configuration for V3-specific engine thresholds and session hours.

## 📂 `agents/` (Data Ingestion)
*   **`data_agent.py`**: Directly communicates with MT5 to download historical M5 and H1 candle data.
*   **`cleaning_agent.py`**: Cleans the raw MT5 data, handles missing bars, and ensures time-series continuity.
*   **`feature_agent.py`**: (Legacy) Used for basic feature engineering.

## 📂 `docs/` & `project_structure/`
*   **`AUDIT.md`**: A technical map of the repository used for safe refactoring and identifying duplicate logic.
*   **`System_Documentation.md`**: The complete V3 system documentation, including the "Lifecycle of a Signal" and the Regime Decision Matrix.
*   **`Step1-Structure.txt`**: A legacy architectural roadmap.

---
*Documentation created for AQRS V3 Maintenance.*