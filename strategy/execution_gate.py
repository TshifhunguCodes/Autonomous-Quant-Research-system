from core.logging_utils import get_logger
import pandas as pd
from strategy.smc_ict_engine import SMCEngine
from pathlib import Path
import numpy as np

logger = get_logger(__name__)

class ExecutionGate:
    @staticmethod
    def evaluate_signal(config, signal):
        """Validates session, regime, and system selection rules."""
        hour = pd.to_datetime(signal["time"]).hour
        quality = signal.get("quality", "MEDIUM")
        state = signal.get("market_state", signal.get("market_regime", signal.get("behavior_label", "UNKNOWN")))
        score = signal.get("confirm_score", 0)

        if getattr(config.live, "relaxed_demo_gate", False):
            system_type = "ALPHA" if signal.get("signal") == "ALPHA" else "FLOW_EXP"
            is_exploratory = system_type != "ALPHA"
            lot_multiplier = 1.0 if system_type == "ALPHA" else getattr(config.regime, "flow_risk_multiplier", 0.5)
            final_lot = max(0.01, config.live.lot * lot_multiplier)
            return True, system_type, final_lot, "RELAXED_DEMO_PASS", is_exploratory
        
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

        # Premium/Discount Gate: Don't buy in Premium, Don't sell in Discount
        direction = signal.get("confirmed_signal", "").upper()
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
