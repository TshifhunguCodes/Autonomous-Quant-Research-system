# AQRS — Autonomous Quant Research System

## Repository State Document

Generated: 2026-05-16  
Asset: XAUUSD (Gold vs USD)  
Base Timeframe: M5 (5-minute candles)  
Development Stage: V3 Active — Phase 2+ (Live Demo Execution & Production Hardening)

---

## 1. Project Overview

**AQRS** is a Python-based algorithmic trading research and execution system for XAUUSD. It combines institutional Smart Money Concepts (SMC/ICT) with dual ALPHA/FLOW signal generation, multi-timeframe context, liquidity analysis, market lifecycle classification, and guarded live demo execution through MetaTrader 5.

**Main Purpose:** Generate high-conviction intraday trade setups from M5 price action, validate them through multiple gate layers, execute on a demo MT5 account, and stream institutional-style alerts to Telegram.

**Currently Traded:** XAUUSD (Gold) only.

**Timeframes Used:**
- M5 — primary execution/research candle
- H1 — higher timeframe bias, exhaustion, lifecycle context
- H4 (derived) — higher timeframe reversal risk, alignment scoring

**Development Stage:** Active live demo execution since late April 2026. The system is in Phase 2 of production hardening — running 24/7 with Telegram monitoring, trailing stops, daily FLOW limits, and adaptive filtering.

---

## 2. Repository Structure

```
AQRS/
├── __init__.py
├── main_v3.py                  # V3 CLI entry point (research/backtest/replay/live)
├── main.py                      # V2 legacy pipeline
├── config/
│   ├── app_config.json          # Primary runtime config (market, risk, SMC, telegram)
│   └── v3_config.py             # V3 config wrapper
├── core/
│   ├── v3_engine.py             # V3 orchestrator — pipelines all engines
│   ├── config.py                # Base config & path registry
│   └── logging_utils.py         # Logging setup
├── engines/
│   ├── behavior_engine.py       # Market classification (TREND/BREAKOUT/CHOPPY...)
│   ├── structure_engine.py      # Swing points, BOS, CHOCH, patterns, retracements
│   ├── zone_engine.py           # Support/resistance, order blocks, FVGs
│   ├── liquidity_engine.py      # Sweeps, stop hunts, fake breakouts, trap detection
│   ├── market_lifecycle_engine.py  # Trend maturity (START→HEALTHY→EXTENDED→EXHAUSTING→REVERSAL)
│   ├── mtf_context_engine.py    # H1/H4 alignment scoring
│   └── dynamic_exit_engine.py   # Exit state classification, trailing, partials
├── systems/
│   ├── alpha_system.py          # Precision sniper — strict ALPHA signals (score ≥75)
│   └── flow_system.py           # Exploratory FLOW signals (score ≥55, 6 setups)
├── strategy/
│   ├── execution_gate.py        # Signal validation gate (20+ rule layers)
│   ├── execution_agent.py       # Live execution orchestrator
│   ├── trade_lifecycle_manager.py  # Post-entry management, BE, partials, force exits
│   ├── trailing_stop.py         # ATR-based trailing stop manager
│   ├── session_filter.py        # Weekend/news session blocking
│   ├── flow_daily_tracker.py    # Daily FLOW trade counter (max 6/day)
│   ├── decision_framework.py    # Continuation/reversal/counter-trend decision logic
│   ├── mt5_bridge.py            # MT5 demo execution guard & order sync
│   ├── smc_ict_engine.py        # SMC/ICT kill zones & imbalance detection
│   └── backend_api.py           # FastAPI dashboard backend
├── risk/
│   └── risk_manager.py          # Dynamic position sizing, stop/TP calculation
├── intelligence/
│   ├── adaptive_filter.py       # ML-based anomaly detection on signals
│   ├── pattern_detector.py      # Candlestick pattern detection
│   ├── unsupervised_model.py    # K-means regime detection
│   └── retrain_pipeline.py      # Auto-retrain ML models during live cycles
├── smart_monitor/
│   ├── smart_monitor.py         # Meta-evaluator for signal quality
│   ├── performance_tracker.py   # Rolling performance metrics
│   ├── quality_scorer.py        # Signal quality scoring
│   └── simple_learner.py        # Lightweight learning model
├── notifications/
│   └── telegram_notifier.py     # Telegram bot — ENTRY/CONTINUATION/EXIT alerts
├── replay/
│   └── replay_engine.py         # Candle-by-candle simulation with equity tracking
├── execution/
│   └── mt5_executor.py          # MT5 order execution layer
├── dashboard/
│   └── streamlit_app.py         # Streamlit monitoring dashboard
├── data/
│   ├── raw/                     # Raw MT5 M5/H1 data
│   ├── clean/                   # Cleaned historical data
│   ├── research/                # Research pipeline CSVs
│   ├── backtest/                # Backtest results (multiple config runs)
│   ├── live/                    # Live execution audit, FLOW daily count
│   └── replay/                  # Replay simulation outputs
├── docs/
│   └── AUDIT.md
└── project_structure/
    └── System_Documentation.md
```

---

## 3. Active Engines

### MarketBehaviorEngine
- **Purpose:** Classifies each M5 candle into a behavior label: `TREND_UP`, `TREND_DOWN`, `RANGE`, `BREAKOUT`, `REVERSAL`, `VOLATILE`, or `CHOPPY`.
- **Inputs:** OHLC, 14-period ATR, 20-period EMA, momentum, candle range, flip count.
- **Outputs:** `behavior_label`, `behavior_confidence` (0–100), feature flags (`trend_up`, `breakout`, `choppy`, etc.).
- **Trading Impact:** ALPHA and FLOW both filter by behavior. CHOPPY and VOLATILE states reduce or block signals.

### PriceActionStructureEngine
- **Purpose:** Detects swing highs/lows, Break of Structure (BOS), Change of Character (CHOCH), double tops/bottoms, break-and-retest patterns, and Fibonacci retracement levels.
- **Inputs:** OHLC, swing logic, prior swings.
- **Outputs:** `structure_state` (HH/HL/LL/LH), `bos`, `choch`, `pattern`, `retracement_class`, `retracement_trade_allowed`.
- **Trading Impact:** Determines trend structure quality. Blocks continuation trades during reversal warnings. Enables ALPHA scoring on BOS/CHOCH/break-retest.

### LiquidityEngine
- **Purpose:** Detects liquidity sweeps, stop hunts, fake breakouts, trap breakouts, and breakout quality.
- **Inputs:** OHLC, BOS, wick/candle body ratios, volatility, exhaustion score, reversal status.
- **Outputs:** `liquidity_event` (LIQUIDITY_SWEEP, STOP_HUNT, FAKE_BREAKOUT, TRAP_BREAKOUT, etc.), `breakout_quality`, `trap_probability`.
- **Trading Impact:** Execution gate blocks trades on FAKE_BREAKOUT, BREAKOUT_REJECTION, high trap probability. ALPHA/FLOW scores penalized.

### MarketLifecycleEngine
- **Purpose:** Classifies trend maturity — from TREND_START through TREND_HEALTHY, TREND_EXTENDED, TREND_EXHAUSTING, to REVERSAL_WATCH and REVERSAL_CONFIRMED. Also detects RANGE_COMPRESSION and BREAKOUT_EXPANSION.
- **Inputs:** ATR slope, momentum weakening, failed continuations, wick rejection, volatility compression.
- **Outputs:** `lifecycle_state`, `exhaustion_score`, `continuation_strength`, `trend_health_score`.
- **Trading Impact:** Trade blocking during exhaustion/reversal, lot size reduction during late trend phases.

### MTFContextEngine
- **Purpose:** Builds H1 and derived-H4 context — bias direction, exhaustion, supply/demand rejection, break-of-structure alignment.
- **Inputs:** H1 OHLC data, M5 direction.
- **Outputs:** `htf_bias` (BULLISH/BEARISH/NEUTRAL), `htf_lifecycle`, `htf_exhaustion`, `multi_tf_alignment_score`, `htf_liquidity_alignment`.
- **Trading Impact:** ALPHA requires alignment ≥65. FLOW requires ≥60. HTF exhaustion blocks aggressive continuations. HTF misalignment reduces scores.

### DynamicExitEngine
- **Purpose:** Determines exit state for open positions based on lifecycle, exhaustion, unrealized R.
- **Inputs:** Lifecycle state, exhaustion, continuation, liquidity event, unrealized R, HTF alignment.
- **Outputs:** `exit_state` (OPEN/PROTECTED/SCALE_ALLOWED/WEAKENING/EXIT_WARNING/FORCE_EXIT), `partial_taken`, `runner_active`, `dynamic_trailing_distance`.
- **Trading Impact:** Controls when positions go to break-even, take partials, tighten stops, or force close. Enables stacking only during SCALE_ALLOWED.

### TradeLifecycleManager
- **Purpose:** Manages open positions — moves stops to break-even, tightens stops on weakening, takes 50% partials, force-closes on exit warnings.
- **Inputs:** Position data, tick, exit plan from DynamicExitEngine.
- **Outputs:** MT5 order modifications, lifecycle events → Telegram alerts.
- **Trading Impact:** Protects profits, enforces discipline, prevents full reversals.

### ExecutionGate
- **Purpose:** Multi-layer signal validation — stale signal, adaptive filter, spread ratio, slippage, kill zone, SMC imbalance, retracement blocks, liquidity rejection, HTF alignment, stacking restrictions, daily FLOW limits, smart monitor.
- **Inputs:** Full signal row, config, market tick.
- **Outputs:** `(allowed, system, lot, reason, is_exploratory)`.
- **Trading Impact:** The most complex gate in the system — over 20 distinct rule layers. Can block trades, reduce lot sizes, or force FLOW treatment.

### ExecutionAgent
- **Purpose:** Orchestrates live execution — loads latest signal, validates freshness, runs session filter, builds trade plan, evaluates gate, sends Telegram alert, places orders, updates trailing stops.
- **Inputs:** Live M5 pipeline output, MT5 tick data.
- **Outputs:** MT5 orders, execution audit log, Telegram alerts.

### TelegramNotifier
- **Purpose:** Sends formatted Markdown trade alerts to Telegram.
- **Inputs:** Config with bot token/chat ID, alert type, signal payload.
- **Outputs:** Telegram messages with symbol, system, side, price, SL/TP, lifecycle, HTF bias, liquidity, scores.

### Analytics Engines (Intelligence/)
- **AdaptiveFilter:** ML anomaly detection on incoming signals — tracks statistical deviations, blocks anomalous trades.
- **PatternDetector:** Identifies candlestick patterns (hammer, engulfing, etc.).
- **UnsupervisedRegimeDetector:** K-means clustering for regime detection.
- **SmartMonitor:** Meta-evaluator that scores signal quality and adjusts lot sizes based on historical performance.

---

## 4. Trading Architecture

### ALPHA Strategy (Precision Sniper)
- **Entry Criteria:** Score ≥ 75. Requires: trending behavior (TREND_UP/DOWN), strong structure (HH/LL), BOS, no fake breakouts, trap probability < 70, MTF alignment ≥ 65, HTF liquidity alignment ≥ 0, HTF exhaustion < 70. Blocked during TREND_EXHAUSTING, REVERSAL_WATCH, FORCE_EXIT.
- **Score Contributors:** Trend (+20), structure (+15), BOS (+15), break-retest (+12), patterns (+10), order blocks (+10), London/NY session (+8).
- **Behavior:** High-conviction, longer holding, wider stops (2.8x ATR), normal lot size.

### FLOW Strategy (Exploratory / Scalper)
- **Entry Criteria:** Score ≥ 55 (configurable, min 45). Has 6 sub-types:
  1. **MOMENTUM_CONTINUATION** — Trend continuation after shallow retracement (20-38% fib).
  2. **MICRO_RETRACEMENT_REENTRY** — Re-entry after 38-50% retracement with rejection.
  3. **EXHAUSTION_FADE** — Fading extreme momentum (overbought/oversold + wick rejection).
  4. **EARLY_REVERSAL_ENTRY** — Early reversal after CHOCH but before full structure break.
- **Score Contributors:** Range/breakout behavior (+18), structure (+12), patterns/setups (+12), order blocks/FVGs (+10), MTF alignment (+10), breakout quality (+8), sweep rejection (+8). Penalized heavily for fake breakout (-35), trap probability (-20), HTF exhaustion (-18).
- **Behavior:** Lower conviction, tighter stops (2.2x ATR), reduced lot size (0.5x), max 3 open simultaneous trades, max 6 per day, 5-minute signal expiry.

### Continuation Logic
- Controlled by `continuation_strength` (0–100) and `retracement_class`.
- **Shallow continuation (0-38% fib):** Highest confidence. ALPHA/FLOW both allow.
- **Normal continuation (38-61.8%):** Standard confidence.
- **Deep continuation (61.8-78.6%):** Requires structure holding (higher lows intact) or blocks.
- **Missing structure anchor during continuation blocks the trade.**

### Reversal Logic
- **Warning stage:** `reversal_warning` when retracement exceeds 78.6% fib but no BOS yet. Blocks continuation trades.
- **Confirmed stage:** `confirmed_reversal` when retracement > 78.6% + BOS + CHOCH + close beyond last swing. Sets lifecycle to REVERSAL_CONFIRMED, forces position exit.
- FLOW allows early reversal entries that ALPHA would never touch.

### Retracement Logic
- `_classify_retracement()` in structure_engine computes fib-based retracement percentage from prior swing.
- Uses fib anchor (swing low → high for uptrends, swing high → low for downtrends).
- `retracement_trade_allowed` flag blocks continuation trades when deep retracements lack structure holding.

### Stacking Logic
- Only allowed during `SCALE_ALLOWED` exit state — which requires: unrealized R ≥ 0.5, continuation strength ≥ 70, trend health ≥ 65, healthy lifecycle.
- `TradeLifecycleManager.can_stack()` checks all open positions must be in SCALE_ALLOWED.
- Stacking is blocked during PROTECTED, WEAKENING, EXIT_WARNING, FORCE_EXIT.

### HTF Alignment Logic
- `multi_tf_alignment_score` (0–100) calculated from: M5 direction matching H1 bias (+25), BOS bias match (+10), H1 breakout support (+10), minus HTF exhaustion (-0.2x), H4 reversal risk (-0.2x), supply/demand rejection (-15 each).
- ALPHA requires ≥ 65. FLOW requires ≥ 60.
- HTF liquidity alignment (-1/0/+1) further gates stacking and aggressive continuations.

### Liquidity Logic
- `liquidity_event` categories flag dangerous market conditions.
- KEY REJECTION TYPES: FAKE_BREAKOUT (breakout attempt, no follow-through, wick rejection), BREAKOUT_REJECTION, TRAP_BREAKOUT, STOP_HUNT, CONFIRMED_SWEEP_REJECTION.
- Execution gate blocks continuation trades on liquidity rejection. FLOW counter-trend is also blocked.
- `trap_probability` further gates trades above 75%.

### Dynamic Exits
- Based on unrealized R (multiple of risk), lifecycle state, continuation strength.
- **OPEN:** No action. → **PROTECTED:** Move to break-even (R ≥ 1.0). → **SCALE_ALLOWED:** Stacking enabled (R ≥ 0.5 + healthy trend). → **WEAKENING:** Tighten stop (0.7x ATR). → **EXIT_WARNING:** Tight aggressively (0.45x ATR) + take 50% partial. → **FORCE_EXIT:** Close position.
- Partial exits: Only once per ticket (tracked via `_partial_exit_tickets` set).
- Runners: Keep 50% position running if continuation ≥ 72, trend health ≥ 60, HTF alignment ≥ 65.

---

## 5. Market States

| State | Meaning | Criteria |
|---|---|---|
| **TREND_START** | Fresh trend initiation | Shallow retracement + continuation ≥ 70 |
| **TREND_HEALTHY** | Trend in good condition | continuation 55–75, low exhaustion |
| **TREND_EXTENDED** | Trend stretched but not reversing | continuation 40–55 |
| **TREND_EXHAUSTING** | Energy fading, reversal possible | exhaustion 55–75 |
| **REVERSAL_WATCH** | Reversal conditions met | exhaustion ≥ 75 or reversal warning |
| **REVERSAL_CONFIRMED** | Structure reversal confirmed | CHOCH + BOS + close beyond prior swing |
| **RANGE_COMPRESSION** | Low volatility, potential breakout | ATR ≤ 85% of 20-period average, compression |
| **BREAKOUT_EXPANSION** | Volatility spike with BOS | BOS + range ≥ 120% of average + ATR expansion |

**Transition Flow:**  
TREND_START → TREND_HEALTHY → TREND_EXTENDED → TREND_EXHAUSTING → REVERSAL_WATCH → REVERSAL_CONFIRMED (or back to TREND_START/HEALTHY)

---

## 6. Current Execution Rules

### Stale Signal Protection
- FLOW signals expire after 5 minutes, ALPHA after 10 minutes.
- Broker timezone offset correction applied (30-min+ difference assumed timezone shift).
- Signal latency calculated from candle close time, not open time.

### Smart Stop-Loss Placement
- ALPHA: 2.8x ATR (wider). FLOW: 2.2x ATR (tighter).
- Structural anchor (swing low/high, support/resistance) + volatility buffer.
- Minimum floor distance: max(ATR stop, spread × 3.5, wick noise × 1.2, 12 points) × volatility buffer.
- Broker minimum stop distance enforced (trade_stops_level + safety buffer).

### Slippage Protection
- `slippage_guard` blocks execution if current price drifted > `max_slippage_points` from research entry price.
- `price_drift_ok` flag computed from: drift ≤ max(spread × 4, ATR × 0.60).

### Spread Filtering
- Dynamic spread ratio based on volatility regime (0.3x for low vol, 0.5x for high vol).
- Trend strength override (1.5x multiplier for continuation ≥ 80).
- FLOW blocked if spread ratio to average > 3.0.

### Stacking Restrictions
- Only during SCALE_ALLOWED state (unrealized R ≥ 0.5, strong continuation, healthy trend).
- FLOW capped at 3 simultaneous open trades.
- FLOW cannot add when existing ALPHA positions in drawdown.

### Reversal Blocking
- REVERSAL_WARNING blocks continuation trades.
- REVERSAL_CONFIRMED → FORCE_EXIT on all positions.
- Counter-trend FLOW only allowed for EXHAUSTION_FADE or EARLY_REVERSAL_ENTRY if CHOCH present.

### FLOW vs ALPHA Coordination
- ALPHA prioritized (signal resolution: ALPHA > FLOW — if both fire, ALPHA wins).
- FLOW limited to 6 trades per day (FlowDailyTracker).
- FLOW risk multiplier = 0.5 × config base lot.
- ALPHA gets 1.0x lot multiplier.
- During REVERSAL_WATCH, FLOW lot reduced by 50%, ALPHA blocked entirely.

---

## 7. Telegram Intelligence System

### Alert Types & Information Sent

**ENTRY_ALERT** — Sent immediately when a signal passes the execution gate.
- Symbol, System (ALPHA/FLOW_EXP), Side (BUY/SELL), Price, SL, TP
- Lifecycle state, HTF Bias
- Liquidity event, Continuation Strength, Exhaustion Score
- Trap Probability, Alpha/Flow scores, MTF Alignment
- Exit State

**CONTINUATION_ALERT** — When an existing position moves to PROTECTED or SCALE_ALLOWED.
- Same fields + ticket number.

**WEAKENING_ALERT** — When position enters weakening state (stop tightened).
- Same fields + new trailing distance.

**PARTIAL_EXIT_ALERT** — When 50% position is closed at R ≥ 1.0.
- Same fields + partial volume.

**REVERSAL_WARNING_ALERT** — When position enters EXIT_WARNING (tight stop, prepare for reversal).
- Same fields + exit confidence score.

**FORCE_EXIT_ALERT** — Position force-closed (REVERSAL_CONFIRMED or extreme conditions).
- Same fields + final price.

System heartbeat messages sent every 120 live cycles (~60 minutes) when no issues detected.

---

## 8. Current Analytics

### Execution Audit CSV
- Path: `data/live/execution_audit.csv`
- Records every live signal evaluation: time, symbol, side, system, regime, setup, lot, price, status (EXECUTED/FAILED/BLOCKED), retcode, comment.
- Used for post-trade analysis and debugging.

### Expectancy Tracking
- `apply_adaptive_learning()` in ExecutionGate reads `data/live/trade_outcomes.csv`.
- Filters trades by current regime context (behavior, market regime, session).
- High PF (>1.5, ≥50 trades) → promote lot multiplier to 1.25x.
- Negative expectancy (< 0, ≥30 trades) → reduce lot to 0.5x.

### Trade Outcome Tracking
- Blocked trades logged to execution audit.
- FLOW daily count tracked via `flow_daily_count.csv`.
- No systematic PnL outcome tracking for individual trades visible in current live data (no trade_outcomes.csv found).

### Adaptive Learning
- Regime-context lookup in trade outcomes.
- Minimum 30 trades before adaptation kicks in.
- Current dataset likely insufficient for most specific contexts (<30 trades per context).

### Regime Memory
- Unsupervised K-means regime detector adds regime labels to pipeline.
- AdaptiveFilter builds anomaly detection baselines from historical signal features.
- SmartMonitor meta-evaluator uses rolling performance windows.

---

## 9. Current Known Problems

### Overtrading
- **FLOW_EXP trades are excessive.** On May 8, 2026, the system executed 40+ FLOW trades in a single day — far exceeding the stated 6/day limit (limit enforcement appears incomplete for the relaxed demo gate).
- Multiple FLOW trades fire on consecutive 5-minute candles in the same direction.

### Excessive FLOW Trades
- FLOW accounts for ~80% of all executed trades, ALPHA ~20%.
- FLOW trades use lower minimum scores (45–55 min) and pass through relaxed gates.
- Several FLOW trades with score < 50 still get executed.

### Poor Continuation During Exhaustion
- Execution gate blocks aggressive continuations during TREND_EXHAUSTING, but FLOW still enters.
- Some trades fire when continuation_strength < 60 and exhaustion > 55.

### Weak Reversals
- FLOW early reversal entries (EARLY_REVERSAL_ENTRY) fire on weak CHOCH without full structure confirmation.
- Multiple reversal trades fail (code 10016) due to stop distance issues.

### Stacking Issues
- Multiple ALPHA trades stack on consecutive candles (e.g., 4 ALPHA buys in 15 minutes on Apr 30).
- Stacking restriction relies on `can_stack()` which checks SCALE_ALLOWED state, but positions may not update fast enough.

### Choppy Market Behavior
- Behavior engine classifies as CHOPPY, but trades still fire under FLOW with choppy_risk_multiplier (0.8x).
- Many trades during CHOPPY periods result in stop-loss issues (code 10016).

### Late Entries
- On May 8, 17:00–20:00 UTC, all trades were blocked by PRICE_DRIFT_REJECTION — excellent guard, but indicates signals are firing too late.
- Signal freshness logic may need tuning.

### Weak Exits
- No trade_outcomes.csv found — no systematic exit PnL tracking.
- Dynamic exits rely on lifecycle state updates which may lag actual price action.

### TODO/FIXME Found
- `strategy/session_filter.py`: `# TODO: Integrate with real economic calendar API`

### Broker Compatibility Issues
- Repeated retcode 10016 (invalid stops) — SL/TP distances not meeting broker minimums.
- Retcode 10027 (invalid volume or insufficient money) — FLOW trades at 0.01 lot sometimes fail on small demo accounts.
- Retcode 10018 (invalid order parameters) — spread issues.

---

## 10. Current Strengths

### Market Structure Understanding
- Sophisticated swing detection (BOS, CHOCH, HH/HL/LL/LH).
- Fibonacci retracement classification with structure holding validation.
- Double top/bottom and break-retest pattern detection.

### Continuation Detection
- Four-tier retracement classification (shallow/normal/deep/reversal warning).
- Continuation strength scoring incorporating multiple signals.
- Aggressive continuation logic with multiple safety gates.

### Liquidity Awareness
- Comprehensive liquidity event classification (7 types).
- Fake breakout / trap breakout detection with quality scoring.
- Stop hunt detection combining sweep + wick rejection + exhaustion.

### HTF Filtering
- Merged H1 + derived H4 context with alignment scoring.
- Supply/demand rejection at H1 level.
- Multi-TF alignment prevents trading against the bigger picture.

### Smart Execution Protection
- Stale signal rejection (5/10 minute expiry).
- Price drift guard against late entries.
- Spread-to-stop ratio with dynamic volatility thresholds.
- Broker minimum distance enforcement.

### Adaptive Risk Logic
- Regime-context adaptive learning (promote/punish based on historical PF).
- Quality-based lot adjustment (ELITE/HIGH/MEDIUM).
- Tick-sizing and broker-rule-aware order preparation.

### Institutional-Style Alerts
- Telegram alerts for every trade lifecycle event.
- Comprehensive alert payload (12+ data fields per alert).
- Heartbeat monitoring during live execution.

---

## 11. Important Files

| File | Role |
|---|---|
| `main_v3.py` | CLI entry point. Runs research/backtest/replay/live modes. MT5 initialization, heartbeat loop, auto-retrain. |
| `core/v3_engine.py` | V3 orchestrator. Chains all engines in sequence. Signal resolution & confidence scoring. |
| `engines/behavior_engine.py` | Market classification (TREND_UP/DOWN/RANGE/BREAKOUT/REVERSAL/VOLATILE/CHOPPY). |
| `engines/structure_engine.py` | Swing analysis, BOS/CHOCH, patterns, fib retracements, reversal detection. |
| `engines/liquidity_engine.py` | Liquidity sweep/fake breakout/trap detection with quality scoring. |
| `engines/market_lifecycle_engine.py` | Trend maturity scoring and lifecycle state classification. |
| `engines/mtf_context_engine.py` | H1/H4 alignment, exhaustion, bias detection. |
| `systems/alpha_system.py` | ALPHA signal generation with strict quality filters. |
| `systems/flow_system.py` | FLOW signal generation with 6 sub-types and decision framework integration. |
| `strategy/execution_gate.py` | Signal validation gate (~400 lines, 20+ rule layers). The most complex file in the system. |
| `strategy/execution_agent.py` | Live execution orchestrator, order preparation, trailing stops. |
| `strategy/trade_lifecycle_manager.py` | Post-entry position management (BE, partials, force exits). |
| `strategy/mt5_bridge.py` | MT5 connection, order execution, trade sync, drawdown checks. |
| `notifications/telegram_notifier.py` | Telegram alert formatting and sending. |
| `strategy/flow_daily_tracker.py` | Daily FLOW trade limits (max 6/day). |
| `config/app_config.json` | Runtime configuration (all tuning parameters, SMC rules, risk settings). |

---

## 12. Recent Major Upgrades

1. **FlowDailyTracker** — Added daily FLOW trade counting and 6/day limit enforcement.
2. **SmartMonitor Integration** — Meta-evaluator for signal quality with lot adjustment.
3. **SessionFilter** — Weekend and high-impact news session blocking (TODO: integrate real economic calendar API).
4. **TrailingStopManager** — ATR-based trailing stop management for open positions.
5. **DynamicSpreadThresholds** — Volatility-based spread ratio thresholds (flexible 0.3x–0.5x).
6. **AdaptiveFilter ML** — Anomaly detection on signal features using statistical baselines.
7. **FLOW Smart Scalper Setups** — 4 sub-types (Momentum Continuation, Micro Retracement Reentry, Exhaustion Fade, Early Reversal Entry).
8. **MTF Context Engine** — H1 → M5 alignment scoring with supply/demand rejection detection.
9. **Relaxed Demo Gate** — `--relaxed-demo-gate` flag for observation mode.
10. **Auto-Retrain Pipeline** — ML models auto-retrain during live cycles (every 500 new candles or 24h).
11. **Flexible NY Session Handling** — Score requirements rather than hard blocks during New York session.
12. **Adaptive Learning** — Regime-context risk promotion/punishment based on historical trade outcomes.

---

## 13. Suggested Next Improvements

### Highest Priority Optimizations

1. **FIX FLOW DAILY LIMIT ENFORCEMENT** — The 6/day limit is not being respected (observed 40+ FLOW trades on single days). The relaxed demo gate may bypass the limit check. The flow tracker needs to be mandatory, not gated.
2. **Reduce FLOW Trade Frequency** — Implement minimum gap between FLOW trades (e.g., 15 minutes between same-direction FLOW entries). Add cooldown timer.
3. **Increase FLOW Minimum Score** — Raise `flow_min_confirm_score` from 45 to 60. Current low-scoring FLOW trades are noise.
4. **Fix Broker Stop Distance Issues** — Many FLOW trades fail with retcode 10016. The stop distance calculation for FLOW (2.2x ATR) is too tight for XAUUSD M5. Increase to 2.5–3.0x ATR.
5. **ALPHA Stacking Restriction** — Multiple ALPHA trades on consecutive candles need a minimum interval (e.g., 3 candles between same-direction ALPHA entries).

### Debugging Focus Areas

1. **Signature of losing trades** — No trade_outcomes.csv currently exists. Start recording individual trade PnL (not just execution audit). This would enable proper adaptive learning.
2. **Analyze FLOW reversal trades** — Many FLOW reversals enter against HTF bias. Track reversal trade PnL separately.
3. **Broker-specific behavior** — Retcode patterns suggest the demo broker has different stop distance requirements than the system expects. Profile ideal SL/TP distances.

### Trade Quality Improvements

1. **Entry Timing Refinement** — Many trades enter in the last minute of the M5 candle. Consider processing signals 30 seconds before candle close.
2. **Minimum Continuation Strength for FLOW** — Add `continuation_strength ≥ 40` floor for ALL FLOW trades (currently not enforced for non-continuation types).
3. **Maximum Daily Trade Hard Limit** — Add system-wide `max_daily_trades` that applies to ALL trades (not just FLOW).

### Performance Bottlenecks

1. **MT5 Bridge Sync** — `sync_closed_trades()` called every cycle is slow. Consider periodic sync (every 10 cycles) instead.
2. **CSV I/O** — Every live cycle reads/writes CSV for trade setups and pipeline. Consider in-memory caching with periodic flush.
3. **H1 Data Reload** — MTF context engine reloads H1 CSV every cycle. Cache in memory.

### Architectural Risks

1. **Config File with Telegram Credentials** — `config/app_config.json` contains live Telegram bot token. Risk of accidental commit.
2. **Single Point of Failure** — `execution_gate.py` is 462 lines with 20+ rule layers. Refactor into composable gate modules.
3. **Relaxed Demo Gate** — When `--relaxed-demo-gate` is active, many safety checks are bypassed. This should have stricter logging and be clearly separated from production gate config.

---

## 14. Current Runtime Commands

### Research
```powershell
python main_v3.py --mode research --output data/research/pipeline.csv
```

### Backtest
```powershell
python main_v3.py --mode backtest --output data/backtest/v3_research_output.csv
python main_v3.py --mode backtest --output data/backtest/v3_research_output.csv --skip-readiness
```

### Replay Mode (Candle-by-Candle Simulation)
```powershell
python main_v3.py --mode replay --replay-max-candles 1000 --output data/replay/replay_decisions.csv
```

### Live Preview (No Orders)
```powershell
python main_v3.py --mode live --run-days 20 --poll-seconds 30 --live-lookback-days 7 --relaxed-demo-gate
```

### Live Demo Execution
```powershell
python main_v3.py --mode live --execute --run-days 20 --poll-seconds 30 --live-lookback-days 7 --relaxed-demo-gate
```

### Dashboard
```powershell
python strategy/start_dashboard.py
# Or manually:
python -m strategy.backend_api
streamlit run strategy/streamlit_app.py
```

### Telegram Test
```powershell
python main_v3.py --test-telegram
```

### Integration / Validation
```powershell
python test_v3_integration.py
python validator_institutional.py
```

---

## 15. Current Performance Snapshot

Based on `data/live/execution_audit.csv` (Apr 26 – May 15, 2026):

### Trade Count
- **Total logged evaluations:** ~200 events
- **EXECUTED trades:** ~160
- **FAILED (broker rejection):** ~15 (mostly retcode 10016 — invalid stops)
- **BLOCKED (gate rejection):** ~25 (PRICE_DRIFT, ADAPTIVE_FILTER anomaly, SPREAD)

### ALPHA vs FLOW Split
- **ALPHA:** ~25 executed trades (~15%)
- **FLOW_EXP:** ~135 executed trades (~85%)
- **ELITE quality:** ~5 ALPHA trades
- **HIGH quality:** ~20 ALPHA trades, ~40 FLOW trades
- **MEDIUM quality:** ~0 ALPHA, ~60 FLOW trades

### Estimated Win Rate
Cannot be precisely determined — no trade outcome CSV exists. However, based on entry quality:
- ALPHA ELITE trades show good structure alignment (score 85-93, MTF alignment high).
- FLOW MEDIUM trades often fire in CHOPPY/RANGE/BREAKOUT regimes with scores 46-63 — expected win rate likely below 50%.
- Most trades entered during established M5 trends visible in audit logs.

### Strongest Sessions
- London open (07:00–09:00 UTC) — highest quality signals, best structure.
- Pre-New York (12:00–14:00 UTC) — good continuation during trend days.

### Weakest Sessions
- Late NY/Post-Close (17:00–22:00 UTC) — many PRICE_DRIFT rejections, choppy behavior.
- Weekend overlap — session filter attempts to block but relaxed gate bypasses.

### Average Holding Behavior
Based on lifecycle states in audit data:
- Most trades enter in BREAKOUT or TREND_UP/DOWN states.
- Dynamic exit engine would classify most as OPEN → some reach PROTECTED.
- Without trade_outcomes.csv, actual hold time and closed PnL is unknown.

### Common Losing Setup Types
1. **FLOW reversal entries against HTF bias** — Score 46-59, no BOS/CHOCH confirmation.
2. **FLOW breakout continuation during CHOPPY regime** — Score 53-63, breakout quality < 60.
3. **FLOW trades with stop distance issues** — Many failures at retcode 10016, stops too tight.
4. **Late-breaking FLOW entries near end of trend** — Exhaustion score > 55, continuation < 50.

> **Note:** A `trade_outcomes.csv` file does not currently exist. Creating one and tracking actual closed PnL by setup type is the single highest-value improvement for performance analysis.