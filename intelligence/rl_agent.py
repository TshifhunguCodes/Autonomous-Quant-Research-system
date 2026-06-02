"""
Q-Learning Reinforcement Learning Agent — Trade Approval via Learned P&L Outcomes
Adapted from ATS_US30_NAS into AQRS

Learns from historical trade outcomes (win/loss) to approve or reject signals.
State: Market conditions at entry time
Action: Approve (1) or Reject (0) the signal
Reward: +1 for profitable trade, -1 for losing trade, 0 for no trade
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import logging
import ast
from collections import defaultdict

logger = logging.getLogger(__name__)


class RLAgent:
    """
    Q-Learning agent for trade approval decisions.
    
    Learns a Q-table mapping (market_state, trend, regime, score_bucket) → action.
    Uses epsilon-greedy exploration during training.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.model_path = Path("intelligence/models/rl_qtable.pkl")
        self.meta_path = Path("intelligence/models/rl_metadata.json")
        
        # Q-table: maps state_tuple → {action: q_value}
        self.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0})  # 0=reject, 1=approve
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2       # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Training tracking
        self.total_trades_seen = 0
        self.wins = 0
        self.losses = 0
        self.recent_accuracy = []  # rolling window of last 100 decisions
        
        # Feature discretization bins
        self.score_bins = [0, 30, 45, 60, 70, 85, 100]
        self.regime_labels = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE", "UNKNOWN", "TREND_UP", "TREND_DOWN", "RANGE", "BREAKOUT", "REVERSAL", "CHOPPY"]
        self.trend_labels = ["BULLISH", "BEARISH", "NEUTRAL", "LONG", "SHORT"]
        self.market_states = ["TRENDING", "RANGING", "CHOPPY", "VOLATILE", "TREND_UP", "TREND_DOWN", "RANGE", "BREAKOUT", "REVERSAL"]
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
    
    def _discretize_state(self, signal_row: dict) -> tuple:
        """
        Convert continuous market conditions into a discrete state tuple.
        
        State components:
        - market_state: TRENDING/RANGING/CHOPPY/VOLATILE
        - trend: BULLISH/BEARISH/NEUTRAL
        - regime: BULL/BEAR/SIDEWAYS/VOLATILE/UNKNOWN
        - score_bucket: binned confirm_score
        """
        market_state = str(signal_row.get("market_state", signal_row.get("behavior_label", "RANGING"))).upper()
        trend = str(signal_row.get("trend", signal_row.get("direction", "NEUTRAL"))).upper()
        regime = str(signal_row.get("regime_label", signal_row.get("market_regime", "UNKNOWN")))
        score = float(signal_row.get("confirm_score", signal_row.get("score", 0)))
        direction = self._normalize_direction(signal_row.get("confirmed_signal", signal_row.get("direction", "")))
        htf_bias = self._normalize_bias(signal_row.get("htf_bias", signal_row.get("h1_bias", "NEUTRAL")))
        aligned = self._alignment_bucket(direction, htf_bias, market_state)
        try:
            visual_zone_score = float(signal_row.get("visual_zone_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            visual_zone_score = 0.0
        setup_type = str(signal_row.get("flow_trade_type", signal_row.get("signal", "NONE"))).upper()
        if visual_zone_score >= 78:
            setup_type = "VISUAL_ZONE"
        if setup_type not in {
            "MOMENTUM_CONTINUATION",
            "MICRO_RETRACEMENT_REENTRY",
            "DEEP_PULLBACK_SCALP",
            "STRUCTURE_RETEST_CONTINUATION",
            "ZONE_REVERSAL_REJECTION",
            "EXHAUSTION_FADE",
            "EARLY_REVERSAL_ENTRY",
            "VISUAL_ZONE",
            "ALPHA",
            "FLOW",
        }:
            setup_type = "OTHER"
        session = str(signal_row.get("session", "UNKNOWN")).upper()
        if session not in {"LONDON", "NEW_YORK", "ASIAN", "EUROPEAN", "US", "UNKNOWN"}:
            session = "OTHER"
        
        # Discretize score
        score_bucket = 0
        for i in range(len(self.score_bins) - 1):
            if self.score_bins[i] <= score < self.score_bins[i + 1]:
                score_bucket = i
                break
        if score >= self.score_bins[-1]:
            score_bucket = len(self.score_bins) - 2
        
        # Normalize strings
        market_state = market_state if market_state in self.market_states else "RANGING"
        trend = trend if trend in self.trend_labels else "NEUTRAL"
        regime = regime.upper()
        regime = regime if regime in self.regime_labels else "UNKNOWN"
        
        return (market_state, trend, regime, score_bucket, direction, aligned, setup_type, session)

    def _normalize_direction(self, value: str) -> str:
        direction = str(value or "").strip().upper()
        if direction in {"BUY", "LONG"}:
            return "LONG"
        if direction in {"SELL", "SHORT"}:
            return "SHORT"
        return "NEUTRAL"

    def _normalize_bias(self, value: str) -> str:
        bias = str(value or "NEUTRAL").strip().upper()
        return {
            "BUY": "BULLISH",
            "LONG": "BULLISH",
            "UP": "BULLISH",
            "SELL": "BEARISH",
            "SHORT": "BEARISH",
            "DOWN": "BEARISH",
        }.get(bias, bias if bias in {"BULLISH", "BEARISH", "NEUTRAL"} else "NEUTRAL")

    def _alignment_bucket(self, direction: str, htf_bias: str, market_state: str) -> str:
        if direction == "NEUTRAL":
            return "NO_DIRECTION"
        if (direction == "LONG" and htf_bias == "BEARISH") or (direction == "SHORT" and htf_bias == "BULLISH"):
            return "HTF_CONFLICT"
        if (direction == "LONG" and market_state == "TREND_DOWN") or (direction == "SHORT" and market_state == "TREND_UP"):
            return "M5_COUNTER_TREND"
        if (direction == "LONG" and htf_bias == "BULLISH") or (direction == "SHORT" and htf_bias == "BEARISH"):
            return "HTF_ALIGNED"
        return "NEUTRAL"
    
    def get_action(self, signal_row: dict, force_approve: bool = True) -> dict:
        """
        Decide whether to approve or reject a signal.
        
        Args:
            signal_row: Dict with signal metadata
            force_approve: If True, always approve (useful for initial exploration)
        
        Returns:
            dict with action, confidence, q_values
        """
        state = self._discretize_state(signal_row)
        
        # If we haven't seen many trades yet, approve to gather data
        if force_approve and self.total_trades_seen < 20:
            return {
                "action": 1,
                "approved": True,
                "confidence": 0.5,
                "state": state,
                "q_values": dict(self.q_table[state]),
                "reason": "exploration_phase",
            }
        
        # Unknown states should not be hard-rejected just because both Q-values
        # are still zero. Let the rule stack decide while the agent gathers data.
        q_values = self.q_table[state]
        if abs(q_values.get(0, 0.0)) < 0.001 and abs(q_values.get(1, 0.0)) < 0.001:
            return {
                "action": 1,
                "approved": True,
                "confidence": 0.35,
                "state": state,
                "q_values": {str(k): round(v, 4) for k, v in q_values.items()},
                "reason": "no_prior_state",
            }

        explored = False

        # Epsilon-greedy: explore or exploit
        if np.random.random() < self.epsilon and self.total_trades_seen < 100:
            # Explore: random action
            action = np.random.choice([0, 1])
            explored = True
        else:
            # Exploit: best known action
            action = max(q_values, key=q_values.get)
        
        # Get confidence from Q-value difference
        q_values = self.q_table[state]
        approve_q = q_values.get(1, 0.0)
        reject_q = q_values.get(0, 0.0)
        confidence = abs(approve_q - reject_q) / (abs(approve_q) + abs(reject_q) + 0.001)
        confidence = min(1.0, confidence)
        
        return {
            "action": action,
            "approved": action == 1,
            "confidence": round(confidence, 3),
            "state": state,
            "q_values": {str(k): round(v, 4) for k, v in q_values.items()},
            "reason": "explore" if explored else "exploit",
        }
    
    def learn(self, signal_row: dict, action: int, reward: float) -> dict:
        """
        Update Q-table based on action taken and reward received.
        
        Args:
            signal_row: Dict with signal metadata
            action: 0 (reject) or 1 (approve)
            reward: +1 for win, -1 for loss, 0 for no trade
        
        Returns:
            dict with learning info
        """
        state = self._discretize_state(signal_row)
        old_q = self.q_table[state][action]
        
        # Bellman equation: Q(s,a) = Q(s,a) + lr * (reward + discount * maxQ(s') - Q(s,a))
        # For simplicity, we use immediate reward (no next state, as trades are terminal)
        td_error = reward - old_q
        new_q = old_q + self.learning_rate * td_error
        
        self.q_table[state][action] = new_q
        
        # Update tracking
        self.total_trades_seen += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        
        self.recent_accuracy.append(1 if reward > 0 else 0)
        if len(self.recent_accuracy) > 100:
            self.recent_accuracy.pop(0)
        
        # Decay epsilon
        if self.total_trades_seen > 20:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Save periodically
        if self.total_trades_seen % 10 == 0:
            self._save()
        
        return {
            "old_q": round(old_q, 4),
            "new_q": round(new_q, 4),
            "td_error": round(td_error, 4),
            "total_trades_seen": self.total_trades_seen,
        }
    
    def learn_from_outcome(self, outcome_row: dict) -> dict:
        """
        Learn from a completed trade outcome (logged to trade_outcomes.csv).
        
        Args:
            outcome_row: Dict with entry_time, pnl, behavior_label, market_regime, etc.
        
        Returns:
            dict with learning info
        """
        pnl = float(outcome_row.get("pnl", 0))
        reward = max(-1.0, min(1.0, pnl))
        if abs(reward) < 0.05:
            reward = 0.0
        
        # Reconstruct state from outcome data
        signal_row = {
            "market_state": outcome_row.get("behavior_label", "RANGING"),
            "trend": outcome_row.get("direction", outcome_row.get("side", outcome_row.get("structure_state", "NEUTRAL"))),
            "regime_label": outcome_row.get("market_regime", "UNKNOWN"),
            "confirm_score": float(outcome_row.get("alpha_score", outcome_row.get("flow_score", outcome_row.get("confirm_score", 0))) or 0),
            "score": float(outcome_row.get("alpha_score", outcome_row.get("flow_score", outcome_row.get("confirm_score", 0))) or 0),
            "confirmed_signal": outcome_row.get("confirmed_signal", outcome_row.get("side", "")),
            "htf_bias": outcome_row.get("htf_bias", "NEUTRAL"),
            "flow_trade_type": outcome_row.get("flow_trade_type", "NONE"),
            "visual_zone_score": outcome_row.get("visual_zone_score", 0),
            "session": outcome_row.get("session", "UNKNOWN"),
        }
        
        return self.learn(signal_row, 1, reward)  # We approved this trade, learn from outcome
    
    def learn_from_blocked(self, signal_row: dict) -> dict:
        """
        Learn from a signal that was blocked (no trade taken).
        Reward is 0 (no gain, no loss).
        """
        return self.learn(signal_row, 0, 0.0)
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        win_rate = (self.wins / (self.wins + self.losses) * 100) if (self.wins + self.losses) > 0 else 0
        recent_win_rate = (sum(self.recent_accuracy) / len(self.recent_accuracy) * 100) if self.recent_accuracy else 0
        
        return {
            "total_trades_seen": self.total_trades_seen,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": round(win_rate, 1),
            "recent_100_win_rate_pct": round(recent_win_rate, 1),
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
        }
    
    def filter_signal(self, signal_row: dict, min_confidence: float = 0.3) -> dict:
        """
        High-level API: evaluate a signal and return whether to trade.
        Used by execution_gate.py.
        
        Args:
            signal_row: Dict with signal metadata
            min_confidence: Minimum confidence to approve
        
        Returns:
            dict with approved (bool), confidence, reason
        """
        decision = self.get_action(signal_row, force_approve=self.total_trades_seen < 20)
        
        approved = decision["approved"] and decision["confidence"] >= min_confidence
        
        return {
            "approved": approved,
            "confidence": decision["confidence"],
            "reason": "rl_approved" if approved else f"rl_rejected_conf={decision['confidence']:.2f}",
            "decision": decision,
        }
    
    def _save(self):
        """Save Q-table and metadata to disk."""
        try:
            # Convert defaultdict to regular dict for serialization
            q_dict = {str(k): v for k, v in self.q_table.items()}
            with open(self.model_path, "wb") as f:
                pickle.dump(q_dict, f)
            
            meta = {
                "total_trades_seen": self.total_trades_seen,
                "wins": self.wins,
                "losses": self.losses,
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
            }
            with open(self.meta_path, "w") as f:
                json.dump(meta, f)
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")
    
    def _load(self):
        """Load Q-table and metadata from disk."""
        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    q_dict = pickle.load(f)
                # Reconstruct defaultdict
                self.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0})
                for k, v in q_dict.items():
                    # Parse string tuple back
                    try:
                        parsed = ast.literal_eval(k)
                        self.q_table[parsed] = v
                    except:
                        pass
                logger.info(f"Loaded RL Q-table with {len(self.q_table)} states")
            
            if self.meta_path.exists():
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                self.total_trades_seen = meta.get("total_trades_seen", 0)
                self.wins = meta.get("wins", 0)
                self.losses = meta.get("losses", 0)
                self.epsilon = meta.get("epsilon", self.epsilon)
                logger.info(f"Loaded RL metadata: {self.total_trades_seen} trades seen")
        except Exception as e:
            logger.warning(f"Could not load RL model (first run?): {e}")
