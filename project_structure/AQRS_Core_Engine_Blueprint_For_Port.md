# AQRS Core Engine Blueprint For Porting

This document is a build blueprint for recreating the AQRS architecture in another trading system. Use it as the instruction layer for an AI engineer/model that must copy the core AQRS operating logic while changing only the market-specific assumptions.

Source system: AQRS V3  
Original focus: XAUUSD, US30/NASDAQ-style institutional movement logic  
Target adaptation: currency pairs or any MT5 instrument

## 1. Core Design Principle

AQRS is not one strategy. It is a full decision stack:

1. Extract market data from MT5.
2. Build a market-intelligence pipeline.
3. Classify behavior, structure, zones, liquidity, lifecycle, indicators, patterns and higher-timeframe context.
4. Generate two independent signal families:
   - ALPHA: strict sniper engine.
   - FLOW: broader exploratory/adaptive engine.
5. Resolve one final signal and direction.
6. Apply risk and trade-plan annotations.
7. Run execution gates and adaptive intelligence filters.
8. Send order to MT5 only if all safety, trend, confirmation and broker-history checks pass.
9. Log every decision and sync closed trades back into the learning layer.

The ported system must preserve this sequence. Do not let execution logic call MT5 before the research pipeline has produced a confirmed signal, trade plan, and gate approval.

## 2. Required Folder/Module Contract

Minimum folders:

- `agents/`: MT5 data extraction and cleaning.
- `core/`: config loading, logging, main orchestrator.
- `engines/`: market intelligence engines.
- `systems/`: ALPHA and FLOW strategy engines.
- `strategy/`: execution gate, execution agent, MT5 bridge, indicators, SMC/ICT helpers, session filters, lifecycle management.
- `intelligence/`: adaptive filter, RL agent, retrain pipeline, regime detector.
- `smart_monitor/`: quality scoring, simple learner, performance tracker.
- `risk/`: position sizing and risk annotation.
- `data/`: raw, clean, research, backtest, replay and live artifacts.
- `config/`: app config and engine thresholds.

Minimum executable entrypoint:

- `main_v3.py` or equivalent with modes:
  - `research`
  - `backtest`
  - `replay`
  - `live`
  - `dashboard`

## 3. Data Extraction Layer

Purpose: collect clean M5 and H1 market data from MT5.

Core functions to implement:

- `resolve_symbol(preferred_symbol)`: find broker-specific symbol variants.
- `get_data(symbol, timeframe, bars)`: get recent bars.
- `get_data_range(symbol, timeframe, start_utc, end_utc)`: get historical range.
- `get_data_range_chunked(...)`: handle long history safely.
- `merge_existing_data(path, df_new)`: append, deduplicate by time, sort.
- `run(config)`: extract M5 and H1 data and save raw CSVs.

Required columns from MT5:

- `time`
- `open`
- `high`
- `low`
- `close`
- `tick_volume`
- `spread`
- `real_volume` if broker provides it

For a currency system, symbol resolution must support suffixes like `EURUSDm`, `EURUSD.a`, `GBPUSD.pro`, etc. Do not hardcode XAUUSD aliases only.

## 4. Main Orchestrator

Core class: `AQRSV3Engine`

Required initialization:

- `MarketBehaviorEngine`
- `PriceActionStructureEngine`
- `ZoneEngine`
- `MarketLifecycleEngine`
- `LiquidityEngine`
- `MTFContextEngine`
- `IndicatorEngine`
- `PatternDetector`
- `UnsupervisedRegimeDetector`
- `AlphaSystem`
- `FlowSystem`
- `RiskManager`
- `ReplayEngine`
- `RetrainPipeline`

Required pipeline order:

```text
raw M5/H1 data
-> behavior.classify_market()
-> structure.build_price_action_structure()
-> zone.build_zones()
-> lifecycle.classify_lifecycle()
-> liquidity.classify_liquidity()
-> indicator_engine.enrich_pipeline()
-> pattern_detector.enrich_pipeline()
-> regime_detector.enrich_pipeline()
-> mtf_context.classify_context()
-> alpha.generate_alpha_setups()
-> flow.generate_flow_setups()
-> _resolve_signals()
-> _apply_mtf_confidence()
-> risk.annotate_trade_risk()
-> _annotate_execution_compatibility()
-> latest signal to execution_agent
```

This ordering matters. Indicators and patterns must exist before ALPHA/FLOW scoring. Higher-timeframe context must exist before final scoring and execution gating.

## 5. Market Behavior Engine

Purpose: classify the current market condition from price action.

Required features:

- `prev_close`
- `momentum`
- `ema20`
- `slope`
- `high_20`
- `low_20`
- `tr`
- `atr14`
- `avg_tr_20`
- `range`
- `range_mean`
- `candle_expansion`
- `volatility`
- `trend_up`
- `trend_down`
- `breakout`
- `reversal`
- `flip_count_10`
- `choppy`
- `behavior_confidence`

Required labels:

- `TREND_UP`
- `TREND_DOWN`
- `BREAKOUT`
- `REVERSAL`
- `VOLATILE`
- `CHOPPY`
- `RANGE`

Trend definition:

- Bullish trend: close above EMA20, EMA20 slope positive, close above close from 5 bars ago.
- Bearish trend: close below EMA20, EMA20 slope negative, close below close from 5 bars ago.

For currencies, ATR and volatility thresholds must be pip-aware. Do not reuse gold point floors directly.

## 6. Structure Engine

Purpose: identify market structure and SMC/ICT entry context.

Required features:

- `swing_high`
- `swing_low`
- `last_swing_high`
- `last_swing_low`
- `prev_swing_high`
- `prev_swing_low`
- `bos_up`
- `bos_down`
- `bos`
- `choch`
- `structure_state`
- `double_top`
- `double_bottom`
- `break_retest`
- `pattern`
- `fib_retracement_pct`
- `retracement_class`
- `retracement_trade_allowed`
- `confirmed_reversal`

Required structure states:

- `HH`
- `HL`
- `LL`
- `LH`
- `NEUTRAL`

Required pattern states:

- `DOUBLE_TOP`
- `DOUBLE_BOTTOM`
- `BREAK_RETEST`
- `CHOCH`
- `NONE`

Trade interpretation:

- `HH/HL` supports buys.
- `LL/LH` supports sells.
- `CHOCH` allows reversal logic only when supported by BOS/liquidity/candle evidence.
- `REVERSAL_WARNING` blocks normal continuation trades.

## 7. Zone, Liquidity And SMC/ICT Layer

Required zone features:

- support/resistance levels.
- demand and supply zones.
- order blocks.
- fair value gaps.
- session levels.
- premium/discount classification.
- zone strength.

Required liquidity features:

- liquidity sweep.
- stop hunt detection.
- sweep rejection.
- fake breakout.
- trap probability.
- breakout quality.
- liquidity alignment.

SMC/ICT logic:

- Buy setups are stronger in discount, demand, bullish order block, bullish FVG, support, or post-sweep rejection.
- Sell setups are stronger in premium, supply, bearish order block, bearish FVG, resistance, or post-sweep rejection.
- Do not buy premium unless the setup is a high-quality breakout/continuation.
- Do not sell discount unless the setup is a confirmed breakdown/reversal.

## 8. Indicator Confirmation Layer

The indicator engine must enrich the pipeline before strategy scoring.

Required indicators:

- MACD:
  - `macd`
  - `macd_signal`
  - `macd_histogram`
  - `macd_crossover`
  - `macd_zero_cross`
  - `macd_slope`
- Bollinger Bands:
  - `bb_middle`
  - `bb_upper`
  - `bb_lower`
  - `bb_width`
  - `bb_position`
  - `bb_squeeze`
  - `bb_touch_upper`
  - `bb_touch_lower`
- ADX:
  - `adx`
  - `adx_plus_di`
  - `adx_minus_di`
  - `adx_strength`
  - `adx_bullish_cross`
  - `adx_bearish_cross`
- Stochastic:
  - `stoch_k`
  - `stoch_d`
  - `stoch_k_slow`
  - `stoch_overbought`
  - `stoch_oversold`
  - `stoch_bullish_cross`
  - `stoch_bearish_cross`

Directional confirmation rule:

- Buy confirmation: MACD histogram or slope positive, DI+ greater than DI-, stochastic K greater than D, close above Bollinger middle.
- Sell confirmation: MACD histogram or slope negative, DI- greater than DI+, stochastic K less than D, close below Bollinger middle.
- Continuation FLOW trades require at least partial indicator confirmation.
- A sell must be blocked when bullish indicator tape is strong unless it is a confirmed reversal setup.
- A buy must be blocked when bearish indicator tape is strong unless it is a confirmed reversal setup.

## 9. Higher Timeframe Context Engine

Purpose: make M5 entries aware of H1/H4 direction and exhaustion.

Required features:

- H1 bias: `BULLISH`, `BEARISH`, `NEUTRAL`
- H1 BOS up/down
- H1 lifecycle
- H1 supply/demand rejection
- H1 exhaustion
- H4 exhaustion
- H4 reversal risk
- `htf_bias`
- `htf_lifecycle`
- `htf_exhaustion`
- `htf_liquidity_alignment`
- `multi_tf_alignment_score`

Rules:

- Buy trades are preferred when M5 direction and H1 bias align bullish.
- Sell trades are preferred when M5 direction and H1 bias align bearish.
- Counter-HTF trades are only allowed for confirmed reversal logic.
- High HTF exhaustion reduces continuation confidence.

## 10. ALPHA System

Purpose: strict high-quality sniper engine.

ALPHA should:

- Trade only high-confluence setups.
- Require strong behavior confidence and score.
- Prefer ELITE quality.
- Use HTF alignment, structure, SMC zones, candles and indicator confirmation.
- Avoid choppy/noisy markets.
- Avoid counter-trend trades unless reversal is fully confirmed.

Required output columns:

- `alpha_signal`
- `alpha_score`
- `alpha_direction`
- `alpha_notes`

Direction contract:

- `alpha_direction = LONG` only if bullish structure/candles/context confirm.
- `alpha_direction = SHORT` only if bearish structure/candles/context confirm.
- No direction means no ALPHA trade.

## 11. FLOW System

Purpose: broader learning/data-generation engine with reduced risk.

FLOW setup types:

- `MOMENTUM_CONTINUATION`
- `MICRO_RETRACEMENT_REENTRY`
- `EXHAUSTION_FADE`
- `EARLY_REVERSAL_ENTRY`
- `NONE`

FLOW must output:

- `flow_signal`
- `flow_score`
- `flow_direction`
- `flow_trade_type`
- `flow_counter_trend_allowed`
- `flow_atr_sl_multiplier`
- `flow_rr_ratio`
- `flow_signal_expiry_minutes`
- `flow_max_open_trades`
- `flow_indicator_score`
- `flow_indicator_confirmations`
- `flow_indicator_conflict`

FLOW rules:

- Continuation buys require trend up, bullish candle/tape/structure, and no bearish reversal conflict.
- Continuation sells require trend down, bearish candle/tape/structure, and no bullish reversal conflict.
- Counter-trend FLOW is allowed only for exhaustion fade or early reversal with CHOCH/reversal/liquidity evidence.
- FLOW score floor should be at least 55 by default.
- FLOW risk must be lower than ALPHA risk.
- FLOW daily cap must be configurable.

## 12. Signal Resolution

The orchestrator must reduce ALPHA and FLOW into one final signal.

Resolution rules:

1. Default `signal = NO_TRADE`.
2. If ALPHA exists, ALPHA owns the candle.
3. Else if FLOW exists, FLOW owns the candle.
4. Resolve direction:
   - ALPHA uses `alpha_direction`.
   - FLOW uses `flow_direction`.
5. Invalid or neutral direction cancels the signal.
6. Convert to execution direction:
   - `LONG -> buy`
   - `SHORT -> sell`
   - otherwise `no_trade`

Required final columns:

- `signal`
- `signal_owner`
- `resolved_direction`
- `confirm_score`
- `quality`
- `confirmed_signal`
- `direction`
- `market_regime`
- `market_state`

Quality labels:

- `ELITE`: confirm score >= 85
- `HIGH`: confirm score >= 70
- `MEDIUM`: confirm score >= 55
- `NONE`: no valid trade

## 13. Risk And Trade Plan Layer

Required risk fields:

- `position_risk_pct`
- `position_risk`
- `stop_distance`
- `entry_price`
- `stop_loss`
- `take_profit`
- `position_size`
- `daily_loss_locked`
- `trade_allowed`

Rules:

- ALPHA risk > FLOW risk.
- Stop distance must respect ATR and broker minimum stop level.
- For currencies, define stop floors in pips, not gold points.
- TP is based on configured RR ratio.
- FLOW may use setup-specific ATR stop multipliers:
  - exhaustion fade: tighter.
  - micro reentry: tight.
  - momentum continuation: medium.
  - early reversal: wider.

## 14. Execution Gate

The execution gate is the final decision authority before order creation.

Required hard checks:

- signal freshness.
- direction confirmation.
- indicator confirmation.
- frequency guard.
- adaptive filter.
- relaxed demo gate if enabled, but never bypass core direction/indicator safety.
- smart stop validity.
- price drift validity.
- daily FLOW cap.
- smart monitor approval.
- minimum framework score.
- spread/cost check.
- HTF alignment.
- RSI/momentum sanity.
- volume sanity.
- priority score.
- counter-trend restrictions.
- lifecycle restrictions.
- SMC premium/discount restriction.
- no duplicate candle trade.
- no excessive open positions.
- stack only when allowed.

Important anti-bug rules:

- Never sell into `TREND_UP` unless the trade is an allowed reversal with CHOCH/BOS/liquidity/candle evidence.
- Never buy into `TREND_DOWN` unless the trade is an allowed reversal with CHOCH/BOS/liquidity/candle evidence.
- Never let FLOW direction fall back to random/default direction.
- Never use `relaxed_demo_gate` to bypass direction, indicator, frequency, or adaptive-history blocks.

## 15. Adaptive Intelligence Layer

The AI layer must act as a filter and risk adjuster, not as an uncontrolled trade generator.

Required components:

- `UnsupervisedRegimeDetector`
- `RLAgent`
- `AdaptiveFilter`
- `RetrainPipeline`
- `SimpleTradeLearner`
- broker-history performance filter

State features for RL/adaptive learning:

- behavior label.
- structure state.
- direction.
- HTF bias.
- setup type.
- session.
- quality score bucket.
- FLOW/ALPHA owner.
- market regime.
- indicator confirmation.
- realized PnL.

Learning artifacts:

- `data/live/execution_audit.csv`
- `data/live/trade_outcomes.csv`
- `data/live/mt5_system_trade_history.csv`
- `intelligence/models/rl_qtable.pkl`
- `intelligence/models/rl_metadata.json`
- `data/ai/simple_model.json`

Learning rules:

- Use broker-closed history as truth.
- Block contexts with enough sample size and poor expectancy.
- Do not overfit tiny samples.
- Positive contexts may receive small risk promotion only after strong sample size.
- Negative contexts should reduce or block, not reverse direction blindly.

## 16. MT5 Execution Bridge

Required responsibilities:

- Initialize and validate MT5.
- Check account/trade permission.
- Resolve symbol.
- Pull tick and symbol info.
- Check existing open positions.
- Check duplicate candle trade.
- Check daily drawdown.
- Prepare MT5 order request.
- Normalize price, SL and TP to tick size/digits.
- Respect broker minimum stop levels.
- Execute order.
- Log execution audit.
- Log blocked trades.
- Sync closed trades.
- Export broker-truth trade history.

Order request fields:

- `action`
- `symbol`
- `volume`
- `type`
- `price`
- `sl`
- `tp`
- `deviation`
- `magic`
- `comment`
- `type_time`
- `type_filling`

Order comment convention:

- ALPHA: `AQ_ALPHA_<QUALITY>`
- FLOW: `AQ_FLOW_EXP_<SETUP_CODE>_<QUALITY>`

Setup codes:

- `MOM`
- `REENT`
- `EXH`
- `REV`
- `FLOW`

## 17. Live Mode Loop

Required sequence:

1. Initialize MT5.
2. Load rolling M5 lookback from MT5.
3. Run full research pipeline on rolling data.
4. Save latest `trade_setups.csv`.
5. Select latest row.
6. If latest row has no `buy` or `sell`, do nothing.
7. Build live trade plan from current tick.
8. Run session filter.
9. Run execution gate.
10. If blocked, log blocked trade.
11. If allowed, send alert and execute order only when `--execute` is enabled.
12. Manage open positions and trailing stops.
13. Sync closed trades and retrain periodically.

## 18. Backtest And Replay Requirements

Backtest must:

- Run the same pipeline as live.
- Use candle path logic for SL/TP resolution.
- Separate ALPHA and FLOW results.
- Export summary/trades/equity/report CSVs.

Replay must:

- Step candle-by-candle.
- Record each decision.
- Make debugging possible for a specific candle/time range.
- Include full pipeline columns in replay output.

## 19. Dashboard/Monitoring Requirements

The dashboard should show:

- active market state.
- latest signal and quality.
- ALPHA vs FLOW performance.
- current trades.
- historical trades.
- chart candles with entries.
- expectancy matrix.
- regime/session performance.
- adaptive filter recommendations.
- broker account status.

## 20. Currency System Adaptation Notes

When porting AQRS from XAUUSD/US30/NASDAQ logic into currencies:

- Replace symbol aliases with currency-pair aliases.
- Use pip/tick-aware stop floors.
- Lower ATR assumptions; currencies move differently from indices/gold.
- Session weighting matters more:
  - London open for EUR/GBP pairs.
  - New York overlap for USD pairs.
  - Asia for JPY/AUD/NZD pairs.
- Spread filters must be pair-specific.
- News blackout matters strongly around CPI, FOMC, NFP, central-bank decisions.
- Avoid copying XAUUSD point constants directly.
- Keep the strategy stack, but recalibrate thresholds using backtest/live history.

Suggested currency configuration fields:

- `symbol`
- `point_size`
- `pip_size`
- `min_stop_pips`
- `max_spread_pips`
- `session_weights`
- `magic_number`
- `allowed_pairs`
- `risk_per_trade`
- `flow_risk_multiplier`
- `flow_daily_limit`
- `news_blackout_minutes`

## 21. Minimum Acceptance Tests

Before live use, the target system must pass:

1. Symbol resolution test.
2. MT5 data extraction test for M5 and H1.
3. Pipeline compile/import test.
4. Research pipeline generation test.
5. Backtest engine test.
6. Replay test.
7. Execution gate unit tests:
   - blocks sell in bullish trend.
   - blocks buy in bearish trend.
   - allows trend-aligned buy/sell.
   - allows reversal only with CHOCH/BOS/liquidity evidence.
8. Smart monitor test.
9. Broker-history export test.
10. Live preview mode test.

## 22. Final Instruction For Another AI Builder

Build the target system as an AQRS-style modular trading engine. Do not build a simple signal bot. The system must produce a full market-intelligence row first, then resolve ALPHA/FLOW, then apply confirmations and adaptive learning, then execute through MT5.

Keep the architecture stable:

```text
MT5 data
-> feature/intelligence pipeline
-> ALPHA/FLOW signal engines
-> signal resolver
-> risk/trade plan
-> execution gate
-> MT5 bridge
-> audit/history
-> adaptive learning
```

Only change the instrument-specific constants, session assumptions, stop/spread floors and symbol handling for the currency market.
