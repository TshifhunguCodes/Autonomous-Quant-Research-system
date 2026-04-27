from core.logging_utils import get_logger
import pandas as pd
from strategy.smc_ict_engine import SMCEngine

logger = get_logger(__name__)

class ExecutionGate:
    @staticmethod
    def evaluate_signal(config, signal):
        """Validates session, regime, and system selection rules."""
        hour = pd.to_datetime(signal["time"]).hour
        quality = signal["quality"]
        state = signal["market_state"]
        score = signal["confirm_score"]
        
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

        # New: Cost Efficiency Gate
        # If current spread is > 40% of our Stop Distance, the trade is mathematically 
        # sub-optimal before it even starts.
        stop_dist = float(signal.get("stop_distance", 1.0))
        current_spread = float(signal.get("spread", 0))
        if stop_dist > 0 and (current_spread / stop_dist) > 0.4:
            return False, "NONE", 0, "COST_TO_REWARD_REJECTION", True
        
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
        final_lot = max(0.01, base_lot * lot_multiplier)
        
        return True, system_type, final_lot, "PASS", is_exploratory