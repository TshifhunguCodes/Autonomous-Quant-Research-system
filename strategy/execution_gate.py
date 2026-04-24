from core.logging_utils import get_logger
import pandas as pd

logger = get_logger(__name__)

class ExecutionGate:
    @staticmethod
    def evaluate_signal(config, signal):
        """Validates session, regime, and system selection rules."""
        hour = pd.to_datetime(signal["time"]).hour
        quality = signal["quality"]
        state = signal["market_state"]
        score = signal["confirm_score"]
        
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