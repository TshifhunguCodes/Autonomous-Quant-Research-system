"""
Adaptive Filter — Combines Unsupervised ML + RL Agent into a Final Execution Gate
Adapted from ATS_US30_NAS into AQRS

This is the final ML layer that sits between signal generation and execution.
It combines:
1. UnsupervisedRegimeDetector: Market regime + anomaly detection
2. RLAgent: Learned trade approval from past P&L outcomes
3. Rule-based overrides: Hard blocks on anomalies, extreme volatility
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class AdaptiveFilter:
    """
    Combined ML filter that evaluates signals before execution.
    
    Flow:
    1. Check market regime (unsupervised): anomaly bars are auto-blocked
    2. Check RL agent: learned approval from past outcomes
    3. Apply rule-based overrides: volatility caps, consecutive loss blocks
    4. Return final decision: APPROVED / BLOCKED / CAUTION
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Lazy import to avoid circular dependencies
        self._regime_detector = None
        self._rl_agent = None
        
        # Tracking consecutive losses for dynamic blocking
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # Block after 3 consecutive losses
        self.trades_since_last_block = 0
        
        # Performance tracking per regime
        self.regime_performance = {}  # {regime: {"wins": N, "losses": N, "total": N}}
        self.broker_history_path = Path("data/live/mt5_system_trade_history.csv")
        self._broker_history_cache = None
        self._broker_history_mtime = None
    
    @property
    def regime_detector(self):
        if self._regime_detector is None:
            from intelligence.unsupervised_model import UnsupervisedRegimeDetector
            self._regime_detector = UnsupervisedRegimeDetector(self.config)
        return self._regime_detector
    
    @property
    def rl_agent(self):
        if self._rl_agent is None:
            from intelligence.rl_agent import RLAgent
            self._rl_agent = RLAgent(self.config)
        return self._rl_agent
    
    def evaluate_signal(self, signal_row: dict, pipeline_df: pd.DataFrame = None) -> dict:
        """
        Evaluate a signal through the full adaptive filter.
        
        Args:
            signal_row: Dict with signal metadata (confirm_score, market_state, trend, etc.)
            pipeline_df: Full pipeline DataFrame (needed for regime detection)
        
        Returns:
            dict with:
                - allowed: bool (True = execute, False = block)
                - filter_verdict: str (APPROVED, BLOCKED, CAUTION)
                - reason: str (why it was blocked/approved)
                - details: dict with per-filter results
        """
        details = {}
        reasons = []
        
        # ====== FILTER 1: Unsupervised Regime + Anomaly ======
        regime_info = self._check_regime(signal_row, pipeline_df)
        details["regime"] = regime_info
        
        if regime_info["blocked"]:
            reasons.append(regime_info["reason"])
        
        # ====== FILTER 2: RL Agent ======
        rl_decision = self._check_rl(signal_row)
        details["rl"] = rl_decision
        
        if rl_decision["blocked"]:
            reasons.append(rl_decision["reason"])
        
        # ====== FILTER 3: Consecutive Loss Guard ======
        loss_guard = self._check_consecutive_losses()
        details["consecutive_losses"] = loss_guard
        
        if loss_guard["blocked"]:
            reasons.append(loss_guard["reason"])
        
        # ====== FILTER 4: Regime Performance ======
        perf_check = self._check_regime_performance(signal_row)
        details["regime_performance"] = perf_check
        
        if perf_check["blocked"]:
            reasons.append(perf_check["reason"])

        # ====== FILTER 5: Broker-Realized Context Performance ======
        broker_check = self._check_broker_context_performance(signal_row)
        details["broker_context_performance"] = broker_check

        if broker_check["blocked"]:
            reasons.append(broker_check["reason"])
        
        # ====== FINAL VERDICT ======
        allowed = len(reasons) == 0
        
        if allowed:
            verdict = "APPROVED"
            reason = "all_filters_passed"
        elif len(reasons) >= 2:
            verdict = "BLOCKED"
            reason = "; ".join(reasons)
        else:
            verdict = "CAUTION"
            reason = reasons[0] if reasons else "unknown"
        
        return {
            "allowed": allowed,
            "filter_verdict": verdict,
            "reason": reason,
            "details": details,
        }
    
    def _check_regime(self, signal_row: dict, pipeline_df: pd.DataFrame = None) -> dict:
        """Check market regime and anomaly status."""
        blocked = False
        reason = ""
        
        regime = str(signal_row.get("regime_label", signal_row.get("market_regime", signal_row.get("behavior_label", "UNKNOWN"))))
        if regime == "UNKNOWN":
            regime = str(signal_row.get("market_regime", signal_row.get("behavior_label", "UNKNOWN")))
        is_anomaly = signal_row.get("is_anomaly", False)
        anomaly_score = float(signal_row.get("anomaly_score", 0))
        
        # Block on anomaly bars (statistical outliers)
        if is_anomaly:
            blocked = True
            reason = f"anomaly_detected_score={anomaly_score:.2f}"
        
        # Block in extreme volatility regimes if score is low
        confirm_score = float(signal_row.get("confirm_score", 0))
        if regime == "VOLATILE" and confirm_score < 60:
            blocked = True
            reason = f"volatile_regime_low_score={confirm_score:.0f}"
        
        return {
            "regime": regime,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 3),
            "blocked": blocked,
            "reason": reason,
        }
    
    def _check_rl(self, signal_row: dict) -> dict:
        """Check RL agent approval."""
        rl_result = self.rl_agent.filter_signal(signal_row, min_confidence=0.3)
        
        return {
            "approved": rl_result["approved"],
            "confidence": rl_result["confidence"],
            "blocked": not rl_result["approved"],
            "reason": rl_result["reason"],
            "q_values": rl_result.get("decision", {}).get("q_values", {}),
        }
    
    def _check_consecutive_losses(self) -> dict:
        """Block trading after N consecutive losses."""
        blocked = False
        reason = ""
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.trades_since_last_block < 3:  # Stay blocked for 3 trades
                blocked = True
                reason = f"consecutive_losses={self.consecutive_losses}_max={self.max_consecutive_losses}"
            else:
                # Reset after cooldown
                self.consecutive_losses = 0
                self.trades_since_last_block = 0
        
        return {
            "consecutive_losses": self.consecutive_losses,
            "blocked": blocked,
            "reason": reason,
        }
    
    def _check_regime_performance(self, signal_row: dict) -> dict:
        """Block trading in regimes where win rate is below threshold."""
        regime = str(signal_row.get("regime_label", signal_row.get("market_regime", signal_row.get("behavior_label", "UNKNOWN"))))
        perf = self.regime_performance.get(regime, {"wins": 0, "losses": 0, "total": 0})
        
        blocked = False
        reason = ""
        
        if perf["total"] >= 5:  # Only check after 5+ trades
            win_rate = perf["wins"] / perf["total"] * 100
            if win_rate < 30:  # Below 30% win rate in this regime
                blocked = True
                reason = f"regime_{regime}_win_rate={win_rate:.0f}%_below_30%"
        
        return {
            "regime": regime,
            "performance": perf,
            "blocked": blocked,
            "reason": reason,
        }

    def _check_broker_context_performance(self, signal_row: dict) -> dict:
        """Use realized MT5 history to block contexts that keep losing live money."""
        history = self._load_broker_history()
        if history.empty:
            return {
                "sample": 0,
                "win_rate": None,
                "expectancy": None,
                "blocked": False,
                "reason": "broker_history_unavailable",
            }

        system = self._signal_system(signal_row)
        side = self._normalize_side(signal_row.get("confirmed_signal", signal_row.get("side", "")))
        family = self._setup_family(signal_row)
        if side not in {"BUY", "SELL"}:
            return {"sample": 0, "win_rate": None, "expectancy": None, "blocked": False, "reason": "no_side"}

        system_col = history.get("system", pd.Series("", index=history.index)).astype(str).str.upper()
        side_col = history.get("side", pd.Series("", index=history.index)).astype(str).str.upper()

        same_system_side = history[(system_col.eq(system)) & (side_col.eq(side))]
        same_comment_col = same_system_side.get("comment", pd.Series("", index=same_system_side.index)).astype(str).str.upper()
        same_family = same_system_side[same_comment_col.str.contains(family, na=False)] if family else same_system_side.iloc[0:0]

        exact_stats = self._performance_stats(same_family)
        side_stats = self._performance_stats(same_system_side)
        selected = exact_stats if exact_stats["sample"] >= 8 else side_stats
        context = "setup" if exact_stats["sample"] >= 8 else "side"

        blocked = False
        reason = ""
        if selected["sample"] >= 8:
            weak_win_rate = selected["win_rate"] < 40.0
            negative_expectancy = selected["expectancy"] <= 0.0
            poor_profit_factor = selected["profit_factor"] < 0.90
            if weak_win_rate and (negative_expectancy or poor_profit_factor):
                blocked = True
                reason = (
                    f"broker_{context}_{system}_{side}_{family}_weak "
                    f"wr={selected['win_rate']:.0f}% exp={selected['expectancy']:.2f} n={selected['sample']}"
                )

        return {
            "system": system,
            "side": side,
            "setup_family": family,
            "context": context,
            "sample": selected["sample"],
            "win_rate": selected["win_rate"],
            "expectancy": selected["expectancy"],
            "profit_factor": selected["profit_factor"],
            "blocked": blocked,
            "reason": reason,
        }

    def _load_broker_history(self) -> pd.DataFrame:
        if not self.broker_history_path.exists():
            return pd.DataFrame()
        try:
            mtime = self.broker_history_path.stat().st_mtime
            if self._broker_history_cache is not None and self._broker_history_mtime == mtime:
                return self._broker_history_cache

            df = pd.read_csv(self.broker_history_path, on_bad_lines="skip", low_memory=False)
            if df.empty or "pnl" not in df.columns:
                self._broker_history_cache = pd.DataFrame()
            else:
                df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
                if "result" not in df.columns:
                    df["result"] = np.where(df["pnl"] > 0, "WIN", np.where(df["pnl"] < 0, "LOSS", "BREAKEVEN"))
                self._broker_history_cache = df[df["pnl"].ne(0.0)].copy()
            self._broker_history_mtime = mtime
            return self._broker_history_cache
        except Exception as exc:
            logger.warning("Broker context performance load failed: %s", exc)
            return pd.DataFrame()

    def _performance_stats(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {"sample": 0, "win_rate": 0.0, "expectancy": 0.0, "profit_factor": 1.0}
        pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        sample = int(len(pnl))
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(abs(pnl[pnl < 0].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
        return {
            "sample": sample,
            "win_rate": (wins / sample * 100.0) if sample else 0.0,
            "expectancy": float(pnl.mean()) if sample else 0.0,
            "profit_factor": profit_factor,
        }

    def _signal_system(self, signal_row: dict) -> str:
        signal = str(signal_row.get("signal", "")).upper()
        flow_type = str(signal_row.get("flow_trade_type", "NONE")).upper()
        return "FLOW_EXP" if signal == "FLOW" or flow_type != "NONE" else "ALPHA"

    def _normalize_side(self, value) -> str:
        side = str(value or "").strip().upper()
        return {"LONG": "BUY", "SHORT": "SELL"}.get(side, side)

    def _setup_family(self, signal_row: dict) -> str:
        flow_type = str(signal_row.get("flow_trade_type", "NONE")).upper()
        if flow_type in {"MOMENTUM_CONTINUATION", "NONE"}:
            return "MOM"
        if flow_type == "MICRO_RETRACEMENT_REENTRY":
            return "REEN"
        if flow_type == "DEEP_PULLBACK_SCALP":
            return "DPB"
        if flow_type == "STRUCTURE_RETEST_CONTINUATION":
            return "SRT"
        if flow_type == "ZONE_REVERSAL_REJECTION":
            return "ZRR"
        if flow_type == "EXHAUSTION_FADE":
            return "EXH"
        if flow_type == "EARLY_REVERSAL_ENTRY":
            return "REV"
        return "FLOW"
    
    def record_outcome(self, signal_row: dict, pnl: float):
        """
        Record trade outcome to update adaptive filter state.
        Called after a trade closes.
        """
        # Update consecutive loss tracker
        if pnl > 0:
            self.consecutive_losses = 0
            self.trades_since_last_block += 1
        elif pnl < 0:
            self.consecutive_losses += 1
            self.trades_since_last_block += 1
        
        # Update regime performance
        regime = str(signal_row.get("regime_label", signal_row.get("market_regime", signal_row.get("behavior_label", "UNKNOWN"))))
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {"wins": 0, "losses": 0, "total": 0}
        
        self.regime_performance[regime]["total"] += 1
        if pnl > 0:
            self.regime_performance[regime]["wins"] += 1
        elif pnl < 0:
            self.regime_performance[regime]["losses"] += 1
        
        # Also feed to RL agent
        outcome_row = {
            "behavior_label": signal_row.get("behavior_label", signal_row.get("market_state", "RANGING")),
            "structure_state": signal_row.get("structure_state", "NEUTRAL"),
            "direction": signal_row.get("direction", signal_row.get("confirmed_signal", signal_row.get("side", "NEUTRAL"))),
            "market_regime": signal_row.get("regime_label", signal_row.get("market_regime", "UNKNOWN")),
            "alpha_score": signal_row.get("alpha_score", signal_row.get("confirm_score", 0)),
            "flow_score": signal_row.get("flow_score", 0),
            "confirmed_signal": signal_row.get("confirmed_signal", signal_row.get("side", "")),
            "htf_bias": signal_row.get("htf_bias", signal_row.get("h1_bias", "NEUTRAL")),
            "flow_trade_type": signal_row.get("flow_trade_type", "NONE"),
            "session": signal_row.get("session", "UNKNOWN"),
            "pnl": pnl,
        }
        self.rl_agent.learn_from_outcome(outcome_row)
    
    def get_stats(self) -> dict:
        """Get filter statistics."""
        return {
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "trades_since_last_block": self.trades_since_last_block,
            "regime_performance": self.regime_performance,
            "rl_stats": self.rl_agent.get_stats(),
        }
    
    def reset_consecutive_losses(self):
        """Manually reset the consecutive loss counter."""
        self.consecutive_losses = 0
        self.trades_since_last_block = 0
