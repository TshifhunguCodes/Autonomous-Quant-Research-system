# AQRS V2: Full System Documentation

## 0. Quick Start Checklist
If you are a human operator or an AI assistant setting this up for the first time:
1.  **Environment:** Ensure Python 3.9+ is installed. Run `pip install pandas MetaTrader5 streamlit fastapi uvicorn`.
2.  **Terminal Sync:** Open MetaTrader 5 and ensure you are logged into your XAUUSD broker.
3.  **Validation:** Run `python strategy/verify_env.py` to confirm terminal and directory readiness.
4.  **Data Generation:** Run `python main.py --mode research --refresh-data`. This builds the system's internal "maps."
5.  **Logic Test:** Run `python strategy/smoke_test.py` to ensure the strategy engine is healthy.
6.  **Dashboard (Control Center):** Run `python strategy/start_dashboard.py` to launch the multi-mode dashboard in your browser.
5.  **Live Guard:** Always run `python main.py --mode live` (Preview mode) before using the `--execute-live` flag.

---

## 1. Vision & Strategic Philosophy
The **Autonomous Quant Research System V2 (AQRS V2)** is built on the concept of **Strategic Decoupling**. Most trading bots fail because they try to "win" in every market environment. AQRS V2 accepts that no single logic works everywhere.

The system operates on two distinct, isolated levels:
1.  **QEAlpha (The Sniper):** Focused strictly on capital preservation. It only enters "Elite" setups during high-liquidity sessions (London/Asia) and stays out of choppy or over-volatile markets.
2.  **QEFlowExp (The Sensor):** Focused on data collection. It trades across all market states (including New York and Choppy) with heavily dampened risk. Its primary goal is to provide a continuous feedback loop of live performance data for future optimization.

This dual-path approach ensures that the system "pays the bills" with Alpha while "funding the future" with Flow data.

---

## 1.1 The "Lifecycle of a Signal" (The Flow)
To understand how data becomes a trade, follow this sequence every 5 minutes:
1.  **Ingestion:** The `data_agent` pulls the last 5,000 M5 bars and 2,000 H1 bars.
2.  **Contextualizing:** The `higher_timeframe_agent` looks at H1. If H1 is bullish, only "Buy" signals are allowed on M5 (H1 Alignment).
3.  **State Detection:** The system calculates ATR and price "flips" to decide if the market is Trending, Ranging, Choppy, or Volatile.
4.  **Signal Scoring:** Technical indicators are weighted to produce a `confirm_score` (0-100).
5.  **Quality Assignment:** Based on the score and market state, the signal is labeled **ELITE**, **HIGH**, or **MEDIUM**.
6.  **The Gatekeeper:** 
    *   If it fits **System A's** strict rules, it executes at full risk.
    *   Otherwise, it checks **System B's** exploratory rules and executes at dampened risk (0.25x).
7.  **MT5 Sync:** Before clicking "send," the system checks if a trade already exists for this candle to prevent duplicates (One-Trade-Per-Candle).

---

## 1.2 Regime Decision Matrix

| Market State | System A (Alpha) Action | System B (Flow) Action | Risk Scaling (Base) |
| :--- | :--- | :--- | :--- |
| **Trending** | Active (Score > 85) | Active | 1.0x (A) / 0.5x (B) |
| **Ranging** | Active (Score > 65) | Active | 1.0x (A) / 0.5x (B) |
| **Choppy** | **HARD BLOCK** | Active (Exploratory) | 0.25x Total |
| **Volatile** | Major Zones Only | Active (Exploratory) | 0.50x Total |

---

## 1.3 New York Session Protocol (Refactor V2.1)
*   **QEAlpha (System A):** **DISABLED**. No trades allowed between 13:00 - 17:59.
*   **QEFlowExp (System B):** **RESTRICTED DATA ENGINE**.
    *   **Logic:** Requires ELITE Quality + Trending/Volatile State (Breakout logic).
    *   **Conviction:** Confirm Score floor raised to 90.
    *   **Risk:** Additional 0.5x dampener (Total 0.25x base exploratory risk).
    *   **Capping:** Maximum 1 trade per NY session to prevent over-trading in volatility.

---

## 2. System Architecture

### 2.1 Directory Structure
*   **`/agents`**: The data pipeline. Responsible for fetching raw MT5 bars, cleaning timestamps, and calculating technical features (ATR, Moving Averages, Momentum).
*   **`/core`**: The backbone. Contains centralized configuration (`config.py`), logging utilities, and the primary pipeline orchestrator.
*   **`/strategy`**: The Brain. Contains the logic for market state classification, signal generation, backtesting, and live execution.
*   **`/data`**: The local database. Separated into `raw`, `research`, `backtest`, `replay`, and `live` to prevent data contamination.

### 2.2 Key Components
*   **`pipeline_transforms.py`**: The logic hub that converts raw price data into actionable trade setups.
*   **`backtesting.py`**: A high-fidelity simulator that accounts for spread, commission, slippage, and complex re-entry logic.
*   **`execution_agent.py`**: The gatekeeper for live trading. It enforces strict safety checks before interacting with MetaTrader 5.

---

## 3. The Strategy Logic ("The Brain")

### 3.1 Market Regime Classification
The system identifies four primary market states based on volatility and price action:
1.  **TRENDING:** Sustained price movement with low "flip" counts.
2.  **RANGING:** Price oscillating within defined boundaries.
3.  **CHOPPY (Low Volatility):** Indecisive price action. High variance, low reward.
4.  **VOLATILE (High Volatility):** Extreme price swings (ATR > 2x average).

### 3.2 Dual-System Execution Model

#### QEAlpha (Strict Alpha - System A)
*   **Objective:** Capital preservation and high win-rate.
*   **Quality:** Requires **ELITE** classification only.
*   **Session:** Restricted to London and Asia sessions.
*   **Filters:** Bypasses "Choppy" and "Volatile" states unless a "Major Zone" (Support/Resistance) is present.
*   **Conviction:** Requires a `confirm_score` of 80-85+ depending on the regime.

#### QEFlowExp (Exploratory Flow - System B)
*   **Objective:** Continuous data generation.
*   **Quality:** Accepts ELITE, HIGH, and MEDIUM setups.
*   **Session:** No restrictions (includes New York and Late Session).
*   **Risk:** Fixed at `Base Lot * Flow Multiplier * 0.5`. This allows the system to "test the waters" without significant drawdown.

---

## 4. Operational Workflows

The system is managed via `main.py` using specific modes.

### Phase 1: Preparation
```bash
# Fetch fresh data and rebuild indicators
python main.py --mode research --refresh-data
```

### Phase 2: Validation
```bash
# Run a comprehensive backtest
python main.py --mode backtest --refresh-data

# Quick check after logic changes (uses cached signals)
python main.py --mode summary --reuse-artifacts
```

### Phase 3: Robustness Testing
```bash
# Stress test the strategy against random market slices
python main.py --mode stress --stress-random-runs 10
```

### Phase 4: Production
```bash
# Preview the current signal (Safe)
python main.py --mode live --reuse-artifacts

# Active trading (Orders will be sent to MT5)
python main.py --mode live --execute-live
```

---

## 5. Risk & Safety Mechanisms

### 5.0 Current Fixes
*   The V3 backtest engine now resolves `ALPHA` and `FLOW` trades using actual candle price paths rather than random binomial outcomes.
*   The replay engine now treats break-even exits as commission-only events, avoiding the previous full-risk loss penalty on `BE` closes.

---

## 5. Risk & Safety Mechanisms

### 5.1 The Elite Paradox Filter
The system monitors `confirm_score`. If a score exceeds **100**, it is flagged as "too perfect," indicating potential exhaustion or price climax, and the trade is blocked.

### 5.2 MT5 Sync Gate
To prevent "double-dipping" or duplicate trades on the same M5 candle, the `execution_agent.py` queries existing MT5 positions. If a trade with the same timestamp exists, the execution path is hard-blocked.

### 5.3 Spread Protection
The system calculates the real-time spread from the MT5 tick. If the spread exceeds the `max_spread_allowed` (typically 25-30 points for XAUUSD), the signal is rejected.

### 5.4 Break-Even (BE) Logic
The system employs adaptive Break-Even triggers:
*   **Volatile Markets:** Locks entry at 1.2R to secure profits early.
*   **Standard Markets:** Locks entry at 1.5R to allow for healthy retracements.

---

## 6. Reporting & Metrics

The system generates several critical artifacts in `data/backtest/`:

1.  **`consolidated_stability_report.csv`**: Measures Win Rate standard deviation and Profit Factor consistency across rolling 7-day windows.
2.  **`system_comparison.csv`**: Provides a side-by-side performance audit of System A vs. System B.
3.  **`system_regime_performance.csv`**: A granular breakdown of how each strategy performs in specific sessions (London vs. NY) and market states (Chop vs. Trend).
4.  **In-Sample (IS) vs. Out-of-Sample (OOS)**: Automatically calculates "Win Rate Degradation." If performance drops by more than 20% in OOS data, a "High Overfitting" warning is issued.

---

## 7. Configuration Management

All parameters are controlled via `config/app_config.json`.
*   **`market`**: Symbol settings, point size, and bar history.
*   **`regime`**: Risk multipliers, session hours, and setup weightings.
*   **`backtest`**: Starting balance, commission, and slippage.
*   **`live`**: Execution thresholds, lot sizes, and allowed qualities.

---

## 8. Summary of Strategy "Guards"

| Guard Name | Purpose | Location |
| :--- | :--- | :--- |
| **H1 Alignment** | Only trades in direction of higher timeframe bias. | `backtesting.py` |
| **NY Guard** | Increases conviction requirements during volatile NY open. | `backtesting.py` |
| **Choppy Filter** | Disables Alpha trades in low-probability environments. | `execution_agent.py` |
| **Exhaustion Filter**| Blocks trades with scores > 100 (climax detection). | `execution_agent.py` |
| **Sync Gate** | Prevents multiple trades on a single candle. | `execution_agent.py` |

---

## 9. Advanced Usage: Replay Engine
The Replay Engine (`--mode replay`) is used for forensic debugging. It simulates the market one candle at a time, allowing you to see exactly *why* the system made a decision at a specific moment in history. It produces a `replay_decisions.csv` for step-by-step audit.

---

## 10. Developer Best Practices
1.  **Data Integrity:** Always use `--refresh-data` when changing feature logic in `feature_agent.py`.
2.  **Iterative Tweak:** Use `--mode summary --reuse-artifacts` to test small parameter changes in seconds.
3.  **Stability First:** Prioritize a low **WR_StdDev** (Win Rate Standard Deviation) over a high total Net PnL. Stability allows for higher compounding in the long run.

---
*Documentation generated for AQRS V2 Maintenance & AI Instruction.*
================================================================================