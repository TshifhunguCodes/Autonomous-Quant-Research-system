from core.logging_utils import get_logger
import pandas as pd
from strategy.smc_ict_engine import SMCEngine
from pathlib import Path
import numpy as np
from strategy.decision_framework import score_trade, select_strategy_mode, should_enter_counter_trend_trade

logger = get_logger(__name__)

class ExecutionGate:
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
        retracement_trade_allowed = bool(signal.get("retracement_trade_allowed", 1))
        confirmed_reversal = bool(signal.get("confirmed_reversal", 0))
        bos = bool(signal.get("bos", 0))
        choch = bool(signal.get("choch", 0))
        lifecycle_state = signal.get("lifecycle_state", "TREND_HEALTHY")
        continuation_strength = float(signal.get("continuation_strength", 0.0))
        liquidity_event = str(signal.get("liquidity_event", "NONE"))
        liquidity_sweep = bool(signal.get("liquidity_sweep", 0))
        fake_breakout = bool(signal.get("fake_breakout", 0))
        breakout_quality = float(signal.get("breakout_quality", 50.0))
        trap_probability = float(signal.get("trap_probability", 0.0))
        stop_hunt_detected = bool(signal.get("stop_hunt_detected", 0))
        htf_bias = str(signal.get("htf_bias", "NEUTRAL"))
        htf_lifecycle = str(signal.get("htf_lifecycle", "UNKNOWN"))
        htf_exhaustion = float(signal.get("htf_exhaustion", 50.0))
        htf_liquidity_alignment = int(signal.get("htf_liquidity_alignment", 0))
        multi_tf_alignment_score = float(signal.get("multi_tf_alignment_score", 50.0))
        smart_stop_ok = bool(signal.get("smart_stop_ok", True))
        price_drift_ok = bool(signal.get("price_drift_ok", True))
        flow_trade_type = str(signal.get("flow_trade_type", "NONE"))
        is_flow_signal = str(signal.get("signal", "")).upper() == "FLOW" or flow_trade_type != "NONE"
        flow_counter_trend_allowed = bool(signal.get("flow_counter_trend_allowed", 0))
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
            if strategy_mode == "BOTH_PAUSED":
                return False, "FLOW_EXP", 0, "FLOW_BOTH_PAUSED", True
            if framework_score < 45:
                return False, "FLOW_EXP", 0, "FLOW_SCORE_BELOW_45", True
            if float(signal.get("spread_ratio_to_average", 1.0) or 1.0) > 2.5:
                return False, "FLOW_EXP", 0, "FLOW_SPREAD_REGIME_BLOCK", True
            flow_counter_trend = (
                (state == "TREND_UP" and direction == "SELL")
                or (state == "TREND_DOWN" and direction == "BUY")
            )
            if flow_counter_trend and should_enter_counter_trend_trade(signal) != "ALLOWED":
                return False, "FLOW_EXP", 0, "FLOW_COUNTER_TREND_BLOCKED", True
            lot_multiplier = getattr(config.regime, "flow_risk_multiplier", 0.5) * 0.5
            if strategy_mode == "FLOW_ONLY":
                lot_multiplier *= 0.8
            if flow_trade_type in ["EXHAUSTION_FADE", "EARLY_REVERSAL_ENTRY"]:
                lot_multiplier *= 0.5
            return True, "FLOW_EXP", max(0.01, config.live.lot * lot_multiplier), "FLOW_FRAMEWORK_PASS", True
        
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
            has_imbalance = bool(signal.get("fvg_bullish", False)) or bool(signal.get("fvg_bearish", False))
            if not has_imbalance:
                return False, "NONE", 0, "ALPHA_SMC_NO_IMBALANCE_REJECTION", True

        # MSS Validation: ALPHA trades MUST have a recent liquidity sweep for high probability
        if quality == "ELITE":
            has_sweep = bool(signal.get("sweep_high", False)) or bool(signal.get("sweep_low", False))
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

        # Task 4: Cost Efficiency Gate (Spread Filter)
        # If current spread is > 40% of our Stop Distance, the trade is mathematically 
        # sub-optimal before it even starts.
        stop_dist = float(signal.get("stop_distance", 1.0))
        current_spread = float(signal.get("spread", 0))
        if stop_dist > 0 and (current_spread / stop_dist) > 0.4:
            return False, "NONE", 0, "COST_TO_REWARD_REJECTION", True
        
        # Task 3 & 4: Adaptive Intelligence Logic
        perf_multiplier = ExecutionGate.apply_adaptive_learning(config, signal)
        if perf_multiplier == 0:
            return False, "NONE", 0, "NEGATIVE_EXPECTANCY_ADAPTIVE_BLOCK", True

        # 1. System Selection logic
        alpha_eligible = (
            quality == "ELITE" and 
            hour in config.regime.alpha_session_hours and 
            state not in ["CHOPPY", "VOLATILE"] and
            not (13 <= hour < 18)  # NY Hard Block
        )
        
        # Trend conviction floor for Alpha
        if alpha_eligible and state == "TRENDING" and score < 85:
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
                if hour != 13 or not bool(signal.get("is_first_breakout", False)) or quality != "ELITE" or score < 90:
                    return False, "FLOW_EXP", 0, "NY_MICRO_STRATEGY_VIOLATION", True
                lot_multiplier *= 0.5

        alpha_regime = signal.get("market_regime", signal.get("behavior_label", "UNKNOWN"))
        if system_type == "FLOW_EXP":
            counter_trend_buy = alpha_regime == "TREND_DOWN" and direction == "BUY"
            counter_trend_sell = alpha_regime == "TREND_UP" and direction == "SELL"
            if counter_trend_buy or counter_trend_sell:
                allowed_counter_types = ["EXHAUSTION_FADE", "EARLY_REVERSAL_ENTRY"]
                if flow_trade_type not in allowed_counter_types and (lifecycle_state != "REVERSAL_CONFIRMED" or not (confirmed_reversal and bos and choch)):
                    return False, system_type, 0, "FLOW_COUNTER_TREND_BLOCKED", is_exploratory
                if flow_trade_type == "EARLY_REVERSAL_ENTRY" and not choch:
                    return False, system_type, 0, "FLOW_EARLY_REVERSAL_NEEDS_CHOCH", is_exploratory
                if flow_trade_type == "EXHAUSTION_FADE" and htf_exhaustion < 55 and trap_probability < 35:
                    return False, system_type, 0, "FLOW_EXHAUSTION_NOT_EXTREME", is_exploratory

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
                is_major = bool(signal.get("major_support", 0)) or bool(signal.get("major_resistance", 0))
                if not is_major or score < 90:
                    return False, system_type, 0, "ALPHA_VOLATILE_REJECTION", is_exploratory

        # Calculate Lot
        base_lot = config.live.lot
        if multi_tf_alignment_score >= 80:
            lot_multiplier *= 1.1
        final_lot = max(0.01, base_lot * lot_multiplier * perf_multiplier)
        
        return True, system_type, final_lot, "PASS", is_exploratory

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
