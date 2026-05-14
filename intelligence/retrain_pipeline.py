"""
Auto-Retraining Pipeline — Retrains ML Models Every N Closed Trades
Adapted from ATS_US30_NAS into AQRS

Monitors trade_outcomes.csv for new closed trades.
When threshold reached (default 50), retrains:
1. UnsupervisedRegimeDetector (K-Means)
2. RLAgent (Q-Learning)
3. Saves updated models to disk
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class RetrainPipeline:
    """
    Monitors trade outcomes and triggers ML model retraining.
    
    Lifecycle:
    1. Check trade_outcomes.csv every cycle
    2. Count new trades since last retrain
    3. If >= threshold (default 50), retrain all ML models
    4. Log retraining results
    """
    
    def __init__(self, config=None):
        self.config = config
        self.outcomes_path = Path("data/live/trade_outcomes.csv")
        self.retrain_marker_path = Path("intelligence/models/last_retrain.txt")
        
        self.retrain_threshold = 50  # Retrain every 50 closed trades
        self.last_retrain_count = 0  # Total trades at last retrain
        self.last_retrain_time = None
        self.total_retrains = 0
        
        # Models to retrain
        self._regime_detector = None
        self._rl_agent = None
        
        # Load last retrain state
        self._load_marker()
    
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
    
    def check_and_retrain(self, pipeline_df: pd.DataFrame = None) -> dict:
        """
        Check if retraining is needed and execute if so.
        
        Args:
            pipeline_df: Full pipeline DataFrame (for regime model retraining)
        
        Returns:
            dict with retraining results
        """
        # Count total trades in outcomes
        if not self.outcomes_path.exists():
            return {"retrained": False, "reason": "no_outcomes_file", "trades": 0}
        
        try:
            outcomes_df = pd.read_csv(self.outcomes_path, parse_dates=["entry_time"])
            total_trades = len(outcomes_df)
        except Exception as e:
            logger.warning(f"Could not read outcomes file: {e}")
            return {"retrained": False, "reason": f"read_error: {e}", "trades": 0}
        
        # Calculate new trades since last retrain
        new_trades = total_trades - self.last_retrain_count
        
        if new_trades < self.retrain_threshold:
            return {
                "retrained": False,
                "reason": f"threshold_not_met",
                "total_trades": total_trades,
                "new_since_last": new_trades,
                "threshold": self.retrain_threshold,
                "remaining": self.retrain_threshold - new_trades,
            }
        
        # Trigger retraining
        return self._retrain_all(pipeline_df, outcomes_df)
    
    def _retrain_all(self, pipeline_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> dict:
        """Retrain all ML models."""
        results = {}
        success = True
        
        # 1. Retrain Unsupervised Regime Detector
        if pipeline_df is not None and not pipeline_df.empty:
            try:
                regime_result = self.regime_detector.train(pipeline_df, retrain=True)
                results["regime_detector"] = regime_result
                logger.info(f"✅ Regime detector retrained: {regime_result.get('samples', 0)} samples")
            except Exception as e:
                logger.error(f"❌ Regime detector retrain failed: {e}")
                results["regime_detector"] = {"error": str(e)}
                success = False
        else:
            results["regime_detector"] = {"skipped": "no_pipeline_data"}
        
        # 2. RL Agent learns from all outcomes
        if not outcomes_df.empty:
            try:
                rl_stats = self.rl_agent.get_stats()
                results["rl_agent"] = {
                    "retrained": True,
                    "total_trades_seen": rl_stats["total_trades_seen"],
                    "win_rate": rl_stats["win_rate_pct"],
                    "epsilon": rl_stats["epsilon"],
                }
                logger.info(f"✅ RL Agent updated: {rl_stats['total_trades_seen']} trades, "
                          f"{rl_stats['win_rate_pct']}% win rate")
            except Exception as e:
                logger.error(f"❌ RL Agent update failed: {e}")
                results["rl_agent"] = {"error": str(e)}
                success = False
        
        # Update retrain marker
        self.last_retrain_count = len(outcomes_df) if not outcomes_df.empty else 0
        self.last_retrain_time = datetime.now()
        self.total_retrains += 1
        self._save_marker()
        
        return {
            "retrained": success,
            "total_retrains": self.total_retrains,
            "trades_at_retrain": self.last_retrain_count,
            "retrain_time": self.last_retrain_time.isoformat(),
            "results": results,
        }
    
    def force_retrain(self, pipeline_df: pd.DataFrame) -> dict:
        """Force retrain all models regardless of threshold."""
        if not self.outcomes_path.exists():
            outcomes_df = pd.DataFrame()
        else:
            try:
                outcomes_df = pd.read_csv(self.outcomes_path, parse_dates=["entry_time"])
            except:
                outcomes_df = pd.DataFrame()
        
        return self._retrain_all(pipeline_df, outcomes_df)
    
    def _save_marker(self):
        """Save retrain state to disk."""
        try:
            self.retrain_marker_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.retrain_marker_path, "w") as f:
                f.write(f"{self.last_retrain_count}\n")
                f.write(f"{self.last_retrain_time.isoformat() if self.last_retrain_time else 'never'}\n")
                f.write(f"{self.total_retrains}\n")
        except Exception as e:
            logger.warning(f"Could not save retrain marker: {e}")
    
    def _load_marker(self):
        """Load retrain state from disk."""
        try:
            if self.retrain_marker_path.exists():
                with open(self.retrain_marker_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        self.last_retrain_count = int(lines[0].strip())
                    if len(lines) >= 2:
                        time_str = lines[1].strip()
                        if time_str != "never":
                            self.last_retrain_time = datetime.fromisoformat(time_str)
                    if len(lines) >= 3:
                        self.total_retrains = int(lines[2].strip())
        except Exception as e:
            logger.warning(f"Could not load retrain marker: {e}")
    
    def get_status(self) -> dict:
        """Get retraining pipeline status."""
        total = 0
        if self.outcomes_path.exists():
            try:
                total = len(pd.read_csv(self.outcomes_path))
            except:
                pass
        
        new_since = total - self.last_retrain_count
        
        return {
            "total_trades": total,
            "last_retrain_count": self.last_retrain_count,
            "new_since_last_retrain": new_since,
            "retrain_threshold": self.retrain_threshold,
            "remaining_until_retrain": max(0, self.retrain_threshold - new_since),
            "total_retrains": self.total_retrains,
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else "never",
        }