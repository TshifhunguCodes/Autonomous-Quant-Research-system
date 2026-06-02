from core.logging_utils import get_logger
import pandas as pd
from strategy.smc_ict_engine import SMCEngine
from pathlib import Path
import numpy as np
from strategy.decision_framework import score_trade, select_strategy_mode, should_enter_counter_trend_trade
from strategy.flow_daily_tracker import get_flow_tracker
from smart_monitor import get_smart_monitor

logger = get_logger(__name__)

class ExecutionGate:
    # Lazy-loaded adaptive filter (initialized once)
    _adaptive_filter = None
    
    @classmethod
    def _get_adaptive_filter(cls):
        """Get or create the AdaptiveFilter singleton."""
        if cls._adaptive_filter is None:
            from intelligence.adaptive_filter import AdaptiveFilter
            cls._adaptive_filter = AdaptiveFilter()
        return cls._adaptive_filter
    
    @classmethod
    def record_trade_outcome(cls, signal: dict, pnl: float):
        """Record trade outcome to adaptive filter for learning."""
        try:
            adapter = cls._get_adaptive_filter()
            adapter.record_outcome(signal, pnl)
        except Exception as e:
            logger.warning(f"Adaptive filter record_outcome error: {e}")
    
    @staticmethod
    def evaluate_signal(config, signal):
        """Validates session, regime, and system selection rules."""
        signal_time = pd.to_datetime(signal["time"])
        hour = signal_time.hour
        quality = signal.get("quality", "MEDIUM")
        state = signal.get("market_state", signal.get("market_regime", signal.get("behavior_label", "UNKNOWN")))
        score = signal.get("confirm_score", 0)
        direction = signal.get("confirmed_signal", "").upper()
        retracement_class = signal.get("retracement_class", "NON_TREND")
        retracement_trade_allowed = ExecutionGate._truthy(signal.get("retracement_trade_allowed", 1))
        confirmed_reversal = ExecutionGate._truthy(signal.get("confirmed_reversal", 0))
        bos = ExecutionGate._truthy(signal.get("bos", 0))
        choch = ExecutionGate._truthy(signal.get("choch", 0))
        lifecycle_state = signal.get("lifecycle_state", "TREND_HEALTHY")
        continuation_strength = float(signal.get("continuation_strength", 0.0))
        liquidity_event = str(signal.get("liquidity_event", "NONE"))
        liquidity_sweep = ExecutionGate._truthy(signal.get("liquidity_sweep", 0))
        fake_breakout = ExecutionGate._truthy(signal.get("fake_breakout", 0))
        breakout_quality = float(signal.get("breakout_quality", 50.0))
        trap_probability = float(signal.get("trap_probability", 0.0))
        stop_hunt_detected = ExecutionGate._truthy(signal.get("stop_hunt_detected", 0))
        htf_bias = ExecutionGate._norm_bias(signal.get("htf_bias", signal.get("h1_bias", "NEUTRAL")))
        htf_lifecycle = str(signal.get("htf_lifecycle", "UNKNOWN"))
        htf_exhaustion = float(signal.get("htf_exhaustion", 50.0))
        htf_liquidity_alignment = int(signal.get("htf_liquidity_alignment", 0))
        multi_tf_alignment_score = float(signal.get("multi_tf_alignment_score", 50.0))
        smart_stop_ok = ExecutionGate._truthy(signal.get("smart_stop_ok", True))
        price_drift_ok = ExecutionGate._truthy(signal.get("price_drift_ok", True))
        flow_trade_type = str(signal.get("flow_trade_type", "NONE"))
        is_flow_signal = str(signal.get("signal", "")).upper() == "FLOW" or flow_trade_type != "NONE"
        flow_counter_trend_allowed = ExecutionGate._truthy(signal.get("flow_counter_trend_allowed", 0))
        max_signal_age_seconds = (5 if is_flow_signal else 10) * 60 # Increased FLOW max_signal_age from 3 to 5 minutes
        current_time = pd.to_datetime(signal.get("current_time", pd.Timestamp.utcnow()))
        if current_time.tzinfo is not None and signal_time.tzinfo is None:
            current_time = current_time.tz_localize(None)
        elif current_time.tzinfo is None and signal_time.tzinfo is not None:
            signal_time = signal_time.tz_localize(None)
        signal_age_seconds = max(0.0, (current_time - signal_time).total_seconds())

        # Broker Time Alignment Correction (Internal Gate)
        if signal_age_seconds > 1800:
            hours_offset = round(signal_age_seconds / 3600)
            signal_age_seconds = abs(signal_age_seconds - (hours_offset * 3600))

        if signal_age_seconds > max_signal_age_seconds:
            return False, "NONE", 0, "STALE_SIGNAL_REJECTION", True

        direction_ok, direction_reason = ExecutionGate._direction_confirmation_ok(signal)
        if not direction_ok:
            return False, "NONE", 0, direction_reason, True

        indicator_ok, indicator_reason = ExecutionGate._indicator_confirmation_ok(signal)
        if not indicator_ok:
            return False, "NONE", 0, indicator_reason, True

        frequency_ok, frequency_reason = ExecutionGate._recent_trade_frequency_ok(signal)
        if not frequency_ok:
            return False, "NONE", 0, frequency_reason, True

        # ===== ADAPTIVE FILTER (ML Layer) =====
        try:
            filter_result = ExecutionGate._get_adaptive_filter().evaluate_signal(signal)
            if not filter_result["allowed"]:
                return False, "NONE", 0, f"ADAPTIVE_FILTER:{filter_result['reason']}", True
        except Exception as e:
            logger.warning(f"Adaptive filter error (non-blocking): {e}")
        # =====================================

        if getattr(config.live, "relaxed_demo_gate", False):
            system_type = "ALPHA" if signal.get("signal") == "ALPHA" else "FLOW_EXP"
            is_exploratory = system_type != "ALPHA"
            lot_multiplier = 1.0 if system_type == "ALPHA" else getattr(config.regime, "flow_risk_multiplier", 0.5)
            final_lot = max(0.01, config.live.lot * lot_multiplier)
            return True, system_type, final_lot, "RELAXED_DEMO_PASS", is_exploratory

        if not smart_stop_ok:
            return False, "NONE", 0, "UNREALISTIC_STOP_REJECTION", True

        if not price_drift_ok:
            return False, "NONE", 0, "PRICE_DRIFT_REJECTION", True

        strategy_mode = str(signal.get("strategy_mode", select_strategy_mode(signal)))
        framework_score = float(signal.get("institutional_trade_score", score_trade(signal)))
        if is_flow_signal:
            # Check FLOW daily limit first
            flow_tracker = get_flow_tracker()
            if flow_tracker.is_limit_reached():
                    return False, "FLOW_EXP", 0, "FLOW_DAILY_LIMIT_REACHED (10 max/day)", True
            
            if strategy_mode == "BOTH_PAUSED":
                return False, "FLOW_EXP", 0, "FLOW_BOTH_PAUSED", True
            
            # Use Smart Monitor for comprehensive evaluation
            smart_monitor = get_smart_monitor()
            smart_allow, smart_quality, smart_lot_mult, smart_reason, smart_tier = \
                smart_monitor.evaluate_signal(signal, "FLOW_EXP")
            
            # Store smart monitor results in signal for logging
            signal['smart_quality_score'] = smart_quality
            signal['smart_quality_tier'] = smart_tier
            signal['smart_lot_multiplier'] = smart_lot_mult
            signal['smart_monitor_reason'] = smart_reason
            
            # If smart monitor blocks the trade, respect it
            if not smart_allow:
                return False, "FLOW_EXP", 0, f"SMART_MONITOR_BLOCK: {smart_reason}", True
            
            # Enhanced minimum score for FLOW (55 instead of 45)
            if framework_score < 55:
                return False, "FLOW_EXP", 0, "FLOW_SCORE_BELOW_55", True
            
            # Flexible spread check for FLOW
            if float(signal.get("spread_ratio_to_average", 1.0) or 1.0) > 3.0:
                return False, "FLOW_EXP", 0, "FLOW_SPREAD_REGIME_BLOCK", True
            
            # Trend alignment requirement for FLOW
            flow_direction = direction
            h1_bias = ExecutionGate._norm_bias(signal.get("htf_bias", signal.get("h1_bias", "NEUTRAL")))
            if h1_bias != "NEUTRAL":
                h1_bullish = h1_bias == "BULLISH"
                local_buy_continuation = (
                    flow_direction == "BUY"
                    and state == "TREND_UP"
                    and flow_trade_type in ["NONE", "MOMENTUM_CONTINUATION", "MICRO_RETRACEMENT_REENTRY", "STRUCTURE_RETEST_CONTINUATION"]
                    and multi_tf_alignment_score >= 45
                    and htf_exhaustion < 80
                )
                local_sell_continuation = (
                    flow_direction == "SELL"
                    and state == "TREND_DOWN"
                    and flow_trade_type in ["NONE", "MOMENTUM_CONTINUATION", "MICRO_RETRACEMENT_REENTRY", "STRUCTURE_RETEST_CONTINUATION"]
                    and multi_tf_alignment_score >= 45
                    and htf_exhaustion < 80
                )
                local_exception = (
                    ExecutionGate._deep_pullback_countertrend_ok(signal)
                    or ExecutionGate._story_countertrend_ok(signal)
                    or ExecutionGate._range_reversion_ok(signal)
                    or ExecutionGate._trap_reversal_ok(signal)
                )
                if flow_direction == "BUY" and not h1_bullish and not (local_buy_continuation or local_exception):
                    return False, "FLOW_EXP", 0, "FLOW_H1_MISALIGNMENT", True
                if flow_direction == "SELL" and h1_bullish and not (local_sell_continuation or local_exception):
                    return False, "FLOW_EXP", 0, "FLOW_H1_MISALIGNMENT", True
            
            # Momentum confirmation (RSI filter)
            rsi = float(signal.get("rsi14", 50.0))
            indicator_score = ExecutionGate._floatish(signal.get("flow_indicator_score", 60.0), 60.0)
            zone_indicator_score = ExecutionGate._floatish(signal.get("zone_indicator_score", 0.0), 0.0)
            if flow_direction == "BUY" and rsi < (30 if flow_trade_type == "DEEP_PULLBACK_SCALP" else 40):
                return False, "FLOW_EXP", 0, "FLOW_RSI_TOO_WEAK_BUY", True
            if flow_direction == "SELL" and rsi > (70 if flow_trade_type == "DEEP_PULLBACK_SCALP" else 60):
                return False, "FLOW_EXP", 0, "FLOW_RSI_TOO_WEAK_SELL", True
            if ExecutionGate._truthy(signal.get("zone_indicator_conflict", 0)) and zone_indicator_score < 55:
                return False, "FLOW_EXP", 0, "FLOW_ZONE_INDICATOR_CONFLICT", True
            
            # Volume confirmation
            volume_avg = float(signal.get("volume_avg_20", 1.0) or 1.0)
            current_volume = float(signal.get("volume", 1.0) or 1.0)
            if volume_avg > 0 and current_volume < volume_avg * 0.7:
                return False, "FLOW_EXP", 0, "FLOW_VOLUME_TOO_LOW", True
            
            # Priority scoring for quality filtering
            trend_alignment = 1.0 if (
                (state == "TREND_UP" and flow_direction == "BUY") or 
                (state == "TREND_DOWN" and flow_direction == "SELL")
            ) else 0.0
            
            momentum_score = min(100, max(0, 100 - abs(rsi - 50) * 2))
            volume_score = min(100, (current_volume / volume_avg) * 100) if volume_avg > 0 else 50
            
            priority_score = (
                framework_score * 0.35 + 
                trend_alignment * 100 * 0.25 + 
                max(indicator_score, zone_indicator_score) * 0.20 +
                momentum_score * 0.1 + 
                volume_score * 0.1
            )
            if ExecutionGate._range_reversion_ok(signal):
                priority_score += 15
            
            # Dynamic threshold based on remaining daily trades
            remaining = flow_tracker.get_remaining_trades()
            if remaining <= 1:
                min_priority = 75  # Very selective when near limit
            elif remaining <= 3:
                min_priority = 65  # Moderately selective
            else:
                min_priority = 55  # Standard threshold
            
            if priority_score < min_priority:
                return False, "FLOW_EXP", 0, f"FLOW_PRIORITY_SCORE_LOW ({priority_score:.0f} < {min_priority})", True
            flow_counter_trend = (
                (state == "TREND_UP" and direction == "SELL")
                or (state == "TREND_DOWN" and direction == "BUY")
            )
            if flow_counter_trend and should_enter_counter_trend_trade(signal) != "ALLOWED" and not (
                ExecutionGate._deep_pullback_countertrend_ok(signal)
                or ExecutionGate._story_countertrend_ok(signal)
                or ExecutionGate._range_reversion_ok(signal)
                or ExecutionGate._trap_reversal_ok(signal)
            ):
                return False, "FLOW_EXP", 0, "FLOW_COUNTER_TREND_BLOCKED", True
            # Apply smart monitor lot multiplier
            lot_multiplier = getattr(config.regime, "flow_risk_multiplier", 0.5) * 0.5
            if strategy_mode == "FLOW_ONLY":
                lot_multiplier *= 0.8
            if flow_trade_type in ["EXHAUSTION_FADE", "EARLY_REVERSAL_ENTRY", "ZONE_REVERSAL_REJECTION"]:
                lot_multiplier *= 0.5
            
            # Apply smart monitor's quality-based lot adjustment
            lot_multiplier *= smart_lot_mult
            
            return True, "FLOW_EXP", max(0.01, config.live.lot * lot_multiplier), f"FLOW_FRAMEWORK_PASS ({smart_reason})", True
        
        # Task 4: Slippage Guard
        # Blocks if current price has moved too far from the research entry price
        entry_price = float(signal.get("entry_price", 0))
        current_price = signal.get("current_tick_price", entry_price)
        max_slippage = getattr(config.market, "max_slippage_points", 10) * config.market.point_size
        if entry_price > 0 and abs(current_price - entry_price) > max_slippage:
            return False, "NONE", 0, "SLIPPAGE_GUARD_REJECTION", True

        # ICT Kill Zone Filter: Enforce high-volume windows for non-ELITE trades
        is_kill_zone = SMCEngine.is_ict_kill_zone(hour)
        if not is_kill_zone and quality != "ELITE":
            return False, "NONE", 0, "OUTSIDE_ICT_KILLZONE_REJECTION", True

        # SMC/ICT Multi-Confluence Gate
      # ELITE (ALPHA) signals MUST have a structural imbalance (FVG).
        # FLOW signals are exploratory and do not require full institutional displacement.
        if config.smc.require_ob_or_fvg and quality == "ELITE":
            has_imbalance = ExecutionGate._truthy(signal.get("fvg_bullish", False)) or ExecutionGate._truthy(signal.get("fvg_bearish", False))
            if not has_imbalance:
                return False, "NONE", 0, "ALPHA_SMC_NO_IMBALANCE_REJECTION", True

        # MSS Validation: ALPHA trades MUST have a recent liquidity sweep for high probability
        if quality == "ELITE":
            has_sweep = ExecutionGate._truthy(signal.get("sweep_high", False)) or ExecutionGate._truthy(signal.get("sweep_low", False))
            if not has_sweep and score < 90:
                return False, "NONE", 0, "ALPHA_SMC_NO_SWEEP_REJECTION", True

        if not retracement_trade_allowed:
            return False, "NONE", 0, "RETRACEMENT_TRADE_BLOCKED", True

        if retracement_class == "REVERSAL_WARNING":
            return False, "NONE", 0, "REVERSAL_WARNING_BLOCK", True

        if fake_breakout:
            return False, "NONE", 0, "FAKE_BREAKOUT_CONTINUATION_BLOCK", True

        liquidity_rejection = liquidity_event in [
            "BREAKOUT_REJECTION",
            "TRAP_BREAKOUT",
            "STOP_HUNT",
            "CONFIRMED_SWEEP_REJECTION",
        ]

        aggressive_continuation = (
            state in ["TREND_UP", "TREND_DOWN"]
            and direction in ["BUY", "SELL"]
            and continuation_strength >= 70
        )
        counter_trend_trade = (
            (state == "TREND_UP" and direction == "SELL")
            or (state == "TREND_DOWN" and direction == "BUY")
        )
        htf_against_direction = (
            (direction == "BUY" and htf_bias == "BEARISH")
            or (direction == "SELL" and htf_bias == "BULLISH")
        )

        if aggressive_continuation and breakout_quality < 60:
            return False, "NONE", 0, "BREAKOUT_CONFIRMATION_REQUIRED", True

        if aggressive_continuation and htf_exhaustion >= 70:
            return False, "NONE", 0, "HTF_EXHAUSTION_CONTINUATION_BLOCK", True

        if aggressive_continuation and htf_liquidity_alignment < 0:
            return False, "NONE", 0, "HTF_LIQUIDITY_REJECTION_BLOCK", True

        if aggressive_continuation and multi_tf_alignment_score < 60:
            return False, "NONE", 0, "HTF_ALIGNMENT_REQUIRED", True

        if aggressive_continuation and lifecycle_state not in ["TREND_HEALTHY", "BREAKOUT_EXPANSION"]:
            return False, "NONE", 0, "AGGRESSIVE_CONTINUATION_LIFECYCLE_BLOCK", True

        if lifecycle_state == "TREND_EXHAUSTING" and direction in ["BUY", "SELL"] and aggressive_continuation:
            return False, "NONE", 0, "TREND_EXHAUSTING_CONTINUATION_BLOCK", True

        if liquidity_rejection and aggressive_continuation:
            return False, "NONE", 0, "LIQUIDITY_REJECTION_CONTINUATION_BLOCK", True

        if counter_trend_trade:
            if not flow_counter_trend_allowed and liquidity_event != "CONFIRMED_SWEEP_REJECTION" and not (liquidity_sweep and stop_hunt_detected and confirmed_reversal):
                return False, "NONE", 0, "COUNTER_TREND_NEEDS_SWEEP_REJECTION", True

        if htf_against_direction and counter_trend_trade and htf_lifecycle != "REVERSAL_WATCH":
            return False, "NONE", 0, "HTF_STRUCTURE_CONFLICT_BLOCK", True

        if lifecycle_state == "EXIT_WARNING" and aggressive_continuation:
            return False, "NONE", 0, "EXIT_WARNING_CONTINUATION_BLOCK", True

        if lifecycle_state == "FORCE_EXIT":
            return False, "NONE", 0, "FORCE_EXIT_BLOCK", True

        # Premium/Discount Gate: Don't buy in Premium, Don't sell in Discount
        if (direction == "BUY" and signal.get("is_premium")) or (direction == "SELL" and signal.get("is_discount")):
            return False, "NONE", 0, "INSTITUTIONAL_PRICING_REJECTION", True
        
        # Volume Spike at POC Gate: Requires a volume spike near the previous session's POC
        if config.smc.require_volume_spike_at_poc:
            if not (signal.get("volume_spike", False) and signal.get("near_prev_poc", False)):
                return False, "NONE", 0, "SMC_NO_VOLUME_SPIKE_AT_POC_REJECTION", True

        # Task 4: Dynamic Cost Efficiency Gate (Flexible Spread Filter)
        # Uses ATR-based dynamic thresholds instead of fixed ratios
        stop_dist = float(signal.get("stop_distance", 1.0))
        current_spread = float(signal.get("spread", 0))
        atr_value = float(signal.get("atr14", 15.0))
        continuation_strength = float(signal.get("continuation_strength", 0.0))
        
        # Dynamic spread ratio based on volatility
        if atr_value > 25:
            max_spread_ratio = 0.5  # High volatility - allow wider spreads
        elif atr_value > 15:
            max_spread_ratio = 0.4  # Medium volatility
        else:
            max_spread_ratio = 0.3  # Low volatility - tighter spreads
        
        # Trend strength override - allow wider spreads for strong trends
        if continuation_strength >= 80:
            max_spread_ratio *= 1.5
        
        if stop_dist > 0 and (current_spread / stop_dist) > max_spread_ratio:
            return False, "NONE", 0, "COST_TO_REWARD_REJECTION", True
        
        # Task 3 & 4: Adaptive Intelligence Logic
        perf_multiplier = ExecutionGate.apply_adaptive_learning(config, signal)
        if perf_multiplier == 0:
            return False, "NONE", 0, "NEGATIVE_EXPECTANCY_ADAPTIVE_BLOCK", True

        # 1. System Selection logic with flexible session handling
        # Flexible NY session - increase score requirements instead of hard blocking
        ny_session_active = 13 <= hour < 18
        ny_session_min_score = 75 if ny_session_active else 0
        
        alpha_eligible = (
            quality == "ELITE" and 
            hour in config.regime.alpha_session_hours and 
            state not in ["CHOPPY", "VOLATILE"] and
            score >= ny_session_min_score  # Flexible NY handling
        )
        
        # Trend conviction floor for Alpha (also flexible during NY)
        min_trend_score = 85 if not ny_session_active else 80
        if alpha_eligible and state == "TRENDING" and score < min_trend_score:
            alpha_eligible = False
            
        if alpha_eligible:
            system_type = "ALPHA"
            is_exploratory = False
            lot_multiplier = 1.0
        else:
            system_type = "FLOW_EXP"
            is_exploratory = True
            lot_multiplier = config.regime.flow_risk_multiplier * 0.5
            
            # Strict NY Protocol for Flow
            if 13 <= hour < 18:
                if hour != 13 or not ExecutionGate._truthy(signal.get("is_first_breakout", False)) or quality != "ELITE" or score < 90:
                    return False, "FLOW_EXP", 0, "NY_MICRO_STRATEGY_VIOLATION", True
                lot_multiplier *= 0.5

        alpha_regime = signal.get("market_regime", signal.get("behavior_label", "UNKNOWN"))
        if system_type == "FLOW_EXP":
            counter_trend_buy = alpha_regime == "TREND_DOWN" and direction == "BUY"
            counter_trend_sell = alpha_regime == "TREND_UP" and direction == "SELL"
            if counter_trend_buy or counter_trend_sell:
                allowed_counter_types = ["EXHAUSTION_FADE", "EARLY_REVERSAL_ENTRY", "DEEP_PULLBACK_SCALP", "ZONE_REVERSAL_REJECTION"]
                if flow_trade_type not in allowed_counter_types and (lifecycle_state != "REVERSAL_CONFIRMED" or not (confirmed_reversal and bos and choch)):
                    return False, system_type, 0, "FLOW_COUNTER_TREND_BLOCKED", is_exploratory
                if flow_trade_type == "EARLY_REVERSAL_ENTRY" and not choch:
                    return False, system_type, 0, "FLOW_EARLY_REVERSAL_NEEDS_CHOCH", is_exploratory
                if flow_trade_type == "EXHAUSTION_FADE" and htf_exhaustion < 55 and trap_probability < 35:
                    return False, system_type, 0, "FLOW_EXHAUSTION_NOT_EXTREME", is_exploratory
                if flow_trade_type == "DEEP_PULLBACK_SCALP" and not ExecutionGate._deep_pullback_countertrend_ok(signal):
                    return False, system_type, 0, "FLOW_DEEP_PULLBACK_NOT_CONFIRMED", is_exploratory
                if flow_trade_type == "ZONE_REVERSAL_REJECTION" and not ExecutionGate._story_countertrend_ok(signal):
                    return False, system_type, 0, "FLOW_ZONE_REVERSAL_NOT_CONFIRMED", is_exploratory

        if lifecycle_state == "REVERSAL_WATCH":
            if system_type == "FLOW_EXP":
                lot_multiplier *= 0.5
            else:
                return False, system_type, 0, "REVERSAL_WATCH_STACK_REDUCTION", is_exploratory

        if liquidity_rejection and system_type == "FLOW_EXP":
            return False, system_type, 0, "LIQUIDITY_REJECTION_STACK_BLOCK", is_exploratory

        if trap_probability >= 75 and aggressive_continuation:
            return False, system_type, 0, "TRAP_PROBABILITY_BLOCK", is_exploratory

        if htf_liquidity_alignment < 0 and system_type == "FLOW_EXP":
            return False, system_type, 0, "HTF_STACKING_REJECTION", is_exploratory

        # 2. General Safety Filters
        if score > 100: # Elite Paradox
            return False, system_type, 0, "EXHAUSTION_CLIMAX_REJECTION", is_exploratory

        if system_type == "ALPHA":
            if state == "CHOPPY" and (quality != "ELITE" or score < 80):
                return False, system_type, 0, "ALPHA_CHOPPY_REJECTION", is_exploratory
            
            if config.regime.adaptive_ny_guard and 13 <= hour <= 20 and score < 75:
                return False, system_type, 0, "NY_GUARD_REJECTION", is_exploratory
                
            if state == "VOLATILE":
                is_major = ExecutionGate._truthy(signal.get("major_support", 0)) or ExecutionGate._truthy(signal.get("major_resistance", 0))
                if not is_major or score < 90:
                    return False, system_type, 0, "ALPHA_VOLATILE_REJECTION", is_exploratory

        # Calculate Lot
        base_lot = config.live.lot
        if multi_tf_alignment_score >= 80:
            lot_multiplier *= 1.1
        final_lot = max(0.01, base_lot * lot_multiplier * perf_multiplier)
        
        return True, system_type, final_lot, "PASS", is_exploratory

    @staticmethod
    def _recent_trade_frequency_ok(signal):
        audit_path = Path("data/live/execution_audit.csv")
        if not audit_path.exists():
            return True, "NO_AUDIT_HISTORY"
        try:
            df = pd.read_csv(audit_path, on_bad_lines="skip", low_memory=False)
            if df.empty or "time" not in df.columns:
                return True, "AUDIT_EMPTY"

            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df[df["time"].notna()].copy()
            if df.empty:
                return True, "AUDIT_NO_TIMES"

            current_time = pd.to_datetime(signal.get("current_time", pd.Timestamp.utcnow())).tz_localize(None)
            recent = df[df["time"] >= current_time - pd.Timedelta(hours=1)].copy()
            if "status" in recent.columns:
                recent = recent[recent["status"].astype(str).str.upper().eq("EXECUTED")]

            direction = str(signal.get("confirmed_signal", "")).upper()
            same_side = recent[recent.get("side", pd.Series("", index=recent.index)).astype(str).str.upper().eq(direction)]
            if len(recent) >= 3:
                return False, "HOURLY_TRADE_FREQUENCY_BLOCK"
            if len(same_side) >= 2:
                return False, "SAME_SIDE_FREQUENCY_BLOCK"
            return True, "FREQUENCY_OK"
        except Exception as e:
            logger.warning(f"Frequency guard error (non-blocking): {e}")
            return True, "FREQUENCY_GUARD_ERROR"

    @staticmethod
    def _norm_bias(value):
        bias = str(value or "NEUTRAL").strip().upper()
        return {
            "LONG": "BULLISH",
            "UP": "BULLISH",
            "BUY": "BULLISH",
            "SHORT": "BEARISH",
            "DOWN": "BEARISH",
            "SELL": "BEARISH",
        }.get(bias, bias if bias in {"BULLISH", "BEARISH", "NEUTRAL"} else "NEUTRAL")

    @staticmethod
    def _truthy(value):
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _floatish(value, default=0.0):
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _direction_confirmation_ok(signal):
        direction = str(signal.get("confirmed_signal", "")).upper()
        if direction not in {"BUY", "SELL"}:
            return False, "NO_DIRECTION_REJECTION"

        htf_bias = ExecutionGate._norm_bias(signal.get("htf_bias", signal.get("h1_bias", "NEUTRAL")))
        state = str(signal.get("market_state", signal.get("market_regime", signal.get("behavior_label", "UNKNOWN")))).upper()
        structure_state = str(signal.get("structure_state", "NEUTRAL")).upper()
        flow_type = str(signal.get("flow_trade_type", "NONE")).upper()
        confirmed_reversal = ExecutionGate._truthy(signal.get("confirmed_reversal", 0))
        choch = ExecutionGate._truthy(signal.get("choch", 0))
        bos = ExecutionGate._truthy(signal.get("bos", 0))
        liquidity_sweep = ExecutionGate._truthy(signal.get("liquidity_sweep", 0)) or str(signal.get("liquidity_event", "")) == "CONFIRMED_SWEEP_REJECTION"
        counter_type = flow_type in {"EXHAUSTION_FADE", "EARLY_REVERSAL_ENTRY", "ZONE_REVERSAL_REJECTION"}
        counter_story_exception = (
            ExecutionGate._deep_pullback_countertrend_ok(signal)
            or ExecutionGate._story_countertrend_ok(signal)
            or ExecutionGate._range_reversion_ok(signal)
            or ExecutionGate._trap_reversal_ok(signal)
        )

        bullish_pattern = ExecutionGate._truthy(signal.get("bullish_reversal", 0)) or ExecutionGate._floatish(signal.get("bullish_pattern_score", 0)) > 0
        bearish_pattern = ExecutionGate._truthy(signal.get("bearish_reversal", 0)) or ExecutionGate._floatish(signal.get("bearish_pattern_score", 0)) > 0
        bullish_structure = structure_state in {"HH", "HL"} or ExecutionGate._truthy(signal.get("bos_up", 0)) or ExecutionGate._truthy(signal.get("demand_zone", 0)) or ExecutionGate._truthy(signal.get("is_support", 0))
        bearish_structure = structure_state in {"LL", "LH"} or ExecutionGate._truthy(signal.get("bos_down", 0)) or ExecutionGate._truthy(signal.get("supply_zone", 0)) or ExecutionGate._truthy(signal.get("is_resistance", 0))

        reversal_exception = counter_type and confirmed_reversal and choch and (bos or liquidity_sweep)

        flow_type = str(signal.get("flow_trade_type", "NONE")).upper()
        is_flow = str(signal.get("signal", "")).upper() == "FLOW" or flow_type != "NONE"
        flow_counter_trend_allowed = ExecutionGate._truthy(signal.get("flow_counter_trend_allowed", 0))
        local_continuation = flow_type in {"MOMENTUM_CONTINUATION", "MICRO_RETRACEMENT_REENTRY", "STRUCTURE_RETEST_CONTINUATION", "NONE"}
        flow_local_buy_exception = (
            is_flow
            and local_continuation
            and direction == "BUY"
            and state == "TREND_UP"
            and (bullish_structure or bullish_pattern)
            and ExecutionGate._floatish(signal.get("multi_tf_alignment_score", 50.0), 50.0) >= 45
            and ExecutionGate._floatish(signal.get("htf_exhaustion", 50.0), 50.0) < 80
        )
        flow_local_sell_exception = (
            is_flow
            and local_continuation
            and direction == "SELL"
            and state == "TREND_DOWN"
            and (bearish_structure or bearish_pattern)
            and ExecutionGate._floatish(signal.get("multi_tf_alignment_score", 50.0), 50.0) >= 45
            and ExecutionGate._floatish(signal.get("htf_exhaustion", 50.0), 50.0) < 80
        )

        if htf_bias == "BULLISH" and direction == "SELL" and not (reversal_exception or flow_local_sell_exception or counter_story_exception or (is_flow and flow_counter_trend_allowed)):
            return False, "HTF_BULLISH_SELL_BLOCK"
        if htf_bias == "BEARISH" and direction == "BUY" and not (reversal_exception or flow_local_buy_exception or counter_story_exception or (is_flow and flow_counter_trend_allowed)):
            return False, "HTF_BEARISH_BUY_BLOCK"
        if state == "TREND_UP" and direction == "SELL" and not (reversal_exception or counter_story_exception):
            return False, "TREND_UP_SELL_BLOCK"
        if state == "TREND_DOWN" and direction == "BUY" and not (reversal_exception or counter_story_exception):
            return False, "TREND_DOWN_BUY_BLOCK"
        close = ExecutionGate._floatish(signal.get("close", 0.0))
        ema20 = ExecutionGate._floatish(signal.get("ema20", close), close)
        slope = ExecutionGate._floatish(signal.get("slope", signal.get("ema_slope", 0.0)))
        momentum = ExecutionGate._floatish(signal.get("momentum", 0.0))
        bullish_local_tape = close > ema20 and (slope > 0 or momentum > 0)
        bearish_local_tape = close < ema20 and (slope < 0 or momentum < 0)
        if is_flow and direction == "SELL" and bullish_local_tape and not (bearish_pattern or reversal_exception or counter_story_exception):
            return False, "FLOW_LOCAL_UPTAPE_SELL_BLOCK"
        if is_flow and direction == "BUY" and bearish_local_tape and not (bullish_pattern or reversal_exception or counter_story_exception):
            return False, "FLOW_LOCAL_DOWNTAPE_BUY_BLOCK"
        if direction == "BUY" and bearish_pattern and not (bullish_pattern or reversal_exception):
            return False, "BEARISH_CANDLE_BUY_BLOCK"
        if direction == "SELL" and bullish_pattern and not (bearish_pattern or reversal_exception):
            return False, "BULLISH_CANDLE_SELL_BLOCK"
        if direction == "BUY" and not (bullish_structure or bullish_pattern or reversal_exception or counter_story_exception):
            return False, "BUY_STRUCTURE_CONFIRMATION_REQUIRED"
        if direction == "SELL" and not (bearish_structure or bearish_pattern or reversal_exception or counter_story_exception):
            return False, "SELL_STRUCTURE_CONFIRMATION_REQUIRED"
        return True, "DIRECTION_CONFIRMED"

    @staticmethod
    def _indicator_confirmation_ok(signal):
        flow_type = str(signal.get("flow_trade_type", "NONE")).upper()
        is_flow = str(signal.get("signal", "")).upper() == "FLOW" or flow_type != "NONE"
        if not is_flow:
            return True, "INDICATOR_CONFIRMATION_NOT_REQUIRED"

        direction = str(signal.get("confirmed_signal", "")).upper()
        if direction not in {"BUY", "SELL"}:
            return False, "FLOW_INDICATOR_NO_DIRECTION"

        indicator_score = max(
            ExecutionGate._floatish(signal.get("flow_indicator_score", 60.0), 60.0),
            ExecutionGate._floatish(signal.get("zone_indicator_score", 0.0), 0.0),
        )
        indicator_conflict = ExecutionGate._truthy(signal.get("flow_indicator_conflict", 0))
        zone_indicator_conflict = ExecutionGate._truthy(signal.get("zone_indicator_conflict", 0))
        continuation_type = flow_type in {"NONE", "MOMENTUM_CONTINUATION", "MICRO_RETRACEMENT_REENTRY", "STRUCTURE_RETEST_CONTINUATION"}
        confirmed_reversal = ExecutionGate._truthy(signal.get("confirmed_reversal", 0))
        choch = ExecutionGate._truthy(signal.get("choch", 0))

        if (indicator_conflict or zone_indicator_conflict) and indicator_score < 55 and not (ExecutionGate._story_countertrend_ok(signal) or ExecutionGate._trap_reversal_ok(signal) or (not continuation_type and confirmed_reversal and choch)):
            return False, "FLOW_INDICATOR_CONFLICT_BLOCK"
        if continuation_type and indicator_score < 40:
            return False, f"FLOW_INDICATOR_CONFIRMATION_LOW ({indicator_score:.0f} < 40)"

        macd_hist = ExecutionGate._floatish(signal.get("macd_histogram", 0.0), 0.0)
        plus_di = ExecutionGate._floatish(signal.get("adx_plus_di", 0.0), 0.0)
        minus_di = ExecutionGate._floatish(signal.get("adx_minus_di", 0.0), 0.0)
        stoch_k = ExecutionGate._floatish(signal.get("stoch_k", 50.0), 50.0)
        stoch_d = ExecutionGate._floatish(signal.get("stoch_d", 50.0), 50.0)

        bullish_indicator_tape = macd_hist > 0 and plus_di >= minus_di and stoch_k >= stoch_d
        bearish_indicator_tape = macd_hist < 0 and minus_di >= plus_di and stoch_k <= stoch_d
        if direction == "SELL" and bullish_indicator_tape and continuation_type:
            return False, "FLOW_BULLISH_INDICATORS_SELL_BLOCK"
        if direction == "BUY" and bearish_indicator_tape and continuation_type:
            return False, "FLOW_BEARISH_INDICATORS_BUY_BLOCK"

        return True, "INDICATORS_CONFIRMED"

    @staticmethod
    def _deep_pullback_countertrend_ok(signal):
        flow_type = str(signal.get("flow_trade_type", "NONE")).upper()
        if flow_type != "DEEP_PULLBACK_SCALP":
            return False

        direction = str(signal.get("confirmed_signal", "")).upper()
        state = str(signal.get("market_state", signal.get("market_regime", signal.get("behavior_label", "UNKNOWN")))).upper()
        if not ((state == "TREND_DOWN" and direction == "BUY") or (state == "TREND_UP" and direction == "SELL")):
            return False

        retracement = ExecutionGate._floatish(signal.get("fib_retracement_pct", 0.0), 0.0)
        indicator_score = max(
            ExecutionGate._floatish(signal.get("flow_indicator_score", 0.0), 0.0),
            ExecutionGate._floatish(signal.get("zone_indicator_score", 0.0), 0.0),
        )
        rsi = ExecutionGate._floatish(signal.get("rsi14", 50.0), 50.0)
        has_reaction = (
            ExecutionGate._truthy(signal.get("wick_rejection", 0))
            or ExecutionGate._truthy(signal.get("bullish_reversal", 0))
            or ExecutionGate._truthy(signal.get("bearish_reversal", 0))
            or ExecutionGate._truthy(signal.get("order_block", 0))
            or ExecutionGate._truthy(signal.get("fvg_zone", 0))
        )
        buy_momentum_ok = direction == "BUY" and 25 <= rsi <= 65
        sell_momentum_ok = direction == "SELL" and 35 <= rsi <= 75
        return bool(50.0 <= retracement <= 78.6 and indicator_score >= 40 and has_reaction and (buy_momentum_ok or sell_momentum_ok))

    @staticmethod
    def _story_countertrend_ok(signal):
        flow_type = str(signal.get("flow_trade_type", "NONE")).upper()
        if flow_type != "ZONE_REVERSAL_REJECTION":
            return False

        direction = str(signal.get("confirmed_signal", "")).upper()
        state = str(signal.get("market_state", signal.get("market_regime", signal.get("behavior_label", "UNKNOWN")))).upper()
        story = str(signal.get("market_story", "NEUTRAL")).upper()
        confidence = ExecutionGate._floatish(signal.get("story_confidence", 0.0), 0.0)
        if confidence < 70:
            return False
        if direction == "BUY":
            return state == "TREND_DOWN" and story.startswith("BULLISH") and (
                ExecutionGate._truthy(signal.get("bullish_zone_rejection", 0))
                or "TRIPLE_BOTTOM" in story
                or "REVERSAL_FROM_ZONE" in story
                or ExecutionGate._trap_reversal_ok(signal)
            )
        if direction == "SELL":
            return state == "TREND_UP" and story.startswith("BEARISH") and (
                ExecutionGate._truthy(signal.get("bearish_zone_rejection", 0))
                or "TRIPLE_TOP" in story
                or "REVERSAL_FROM_ZONE" in story
                or ExecutionGate._trap_reversal_ok(signal)
            )
        return False

    @staticmethod
    def _trap_reversal_ok(signal):
        direction = str(signal.get("confirmed_signal", "")).upper()
        trap_direction = str(signal.get("trap_reversal_direction", "NEUTRAL")).upper()
        trap_score = ExecutionGate._floatish(signal.get("trap_reversal_score", 0.0), 0.0)
        visual_score = ExecutionGate._floatish(signal.get("visual_zone_score", 0.0), 0.0)
        zone_score = ExecutionGate._floatish(signal.get("zone_indicator_score", 0.0), 0.0)
        if direction == "BUY":
            expected = "LONG"
        elif direction == "SELL":
            expected = "SHORT"
        else:
            return False
        return bool(
            trap_direction == expected
            and trap_score >= 55
            and max(visual_score, zone_score) >= 55
            and str(signal.get("flow_trade_type", "NONE")).upper() == "ZONE_REVERSAL_REJECTION"
        )

    @staticmethod
    def _range_reversion_ok(signal):
        direction = str(signal.get("confirmed_signal", "")).upper()
        if direction not in {"BUY", "SELL"}:
            return False

        state = str(signal.get("market_state", signal.get("market_regime", signal.get("behavior_label", "UNKNOWN")))).upper()
        if state not in {"RANGE", "CHOPPY"}:
            return False

        indicator_score = max(
            ExecutionGate._floatish(signal.get("flow_indicator_score", 0.0), 0.0),
            ExecutionGate._floatish(signal.get("zone_indicator_score", 0.0), 0.0),
        )
        story_confidence = ExecutionGate._floatish(signal.get("story_confidence", 0.0), 0.0)
        confirm_score = ExecutionGate._floatish(signal.get("confirm_score", signal.get("flow_score", 0.0)), 0.0)
        if max(indicator_score, story_confidence, confirm_score) < 55:
            return False

        bullish_reaction = (
            ExecutionGate._truthy(signal.get("bullish_zone_rejection", 0))
            or ExecutionGate._truthy(signal.get("bullish_reversal", 0))
            or ExecutionGate._floatish(signal.get("bullish_pattern_score", 0.0), 0.0) > 0
            or ExecutionGate._truthy(signal.get("demand_zone", 0))
            or ExecutionGate._truthy(signal.get("is_support", 0))
            or str(signal.get("market_story", "")).upper().startswith("BULLISH")
        )
        bearish_reaction = (
            ExecutionGate._truthy(signal.get("bearish_zone_rejection", 0))
            or ExecutionGate._truthy(signal.get("bearish_reversal", 0))
            or ExecutionGate._floatish(signal.get("bearish_pattern_score", 0.0), 0.0) > 0
            or ExecutionGate._truthy(signal.get("supply_zone", 0))
            or ExecutionGate._truthy(signal.get("is_resistance", 0))
            or str(signal.get("market_story", "")).upper().startswith("BEARISH")
        )
        if direction == "BUY":
            return bullish_reaction
        return bearish_reaction

    @staticmethod
    def apply_adaptive_learning(config, signal):
        """Calculates risk multiplier based on historical outcome expectancy."""
        outcomes_path = Path("data/live/trade_outcomes.csv")
        if not outcomes_path.exists():
            return 1.0

        try:
            df = pd.read_csv(outcomes_path)
            if len(df) < 30: return 1.0 # Minimum statistical floor for adaptive learning

            # Define the current regime context for lookup
            current_behavior = signal.get("behavior_label", "UNKNOWN")
            current_regime = signal.get("market_regime", "NEUTRAL")
            current_session = signal.get("session", "UNKNOWN")

            # Filter for current regime context
            context = df[
                (df["behavior_label"] == current_behavior) &
                (df["market_regime"] == current_regime) &
                (df["session"] == current_session)
            ]

            if context.empty or len(context) < 10: # Need at least 10 trades in this specific context
                return 1.0

            net_pnl = context["pnl"].sum()
            trades = len(context)
            gross_profit = context[context["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(context[context["pnl"] < 0]["pnl"].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0) # Handle zero gross_loss

            # Task 3: Auto-reduce for negative expectancy
            if net_pnl < 0 and trades >= 30: # Require more samples for reduction
                logger.warning("📉 Adaptive Learning: Reducing risk for %s/%s/%s due to negative expectancy (PF: %.2f, Trades: %d).", 
                               current_behavior, current_regime, current_session, pf, trades)
                return 0.5

            # Task 4: Auto-promote high-performers
            if pf > 1.5 and trades >= 50: # Higher PF and more samples for promotion
                logger.info("🚀 Adaptive Learning: Promoting risk for %s/%s/%s (PF: %.2f, Trades: %d).", 
                            current_behavior, current_regime, current_session, pf, trades)
                return 1.25

            return 1.0
        except Exception:
            return 1.0
