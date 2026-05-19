"""
Auto-retraining pipeline for regime and reinforcement-learning models.

The pipeline monitors data/live/trade_outcomes.csv. Once enough newly closed
trades are available, it retrains the unsupervised regime detector and replays
the new outcomes into the RL approval agent.
"""
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class RetrainPipeline:
    """Monitor closed trade outcomes and trigger intelligence updates."""

    def __init__(self, config=None):
        self.config = config
        self.outcomes_path = Path("data/live/trade_outcomes.csv")
        self.retrain_marker_path = Path("intelligence/models/last_retrain.txt")

        self.retrain_threshold = 50
        self.last_retrain_count = 0
        self.last_retrain_time = None
        self.total_retrains = 0

        self._regime_detector = None
        self._rl_agent = None
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
        """Retrain when enough new closed outcomes have accumulated."""
        if not self.outcomes_path.exists():
            return {"retrained": False, "reason": "no_outcomes_file", "trades": 0}

        try:
            outcomes_df = pd.read_csv(self.outcomes_path, parse_dates=["entry_time"], on_bad_lines="skip")
            total_trades = len(outcomes_df)
        except Exception as e:
            logger.warning("Could not read outcomes file: %s", e)
            return {"retrained": False, "reason": f"read_error: {e}", "trades": 0}

        new_trades = total_trades - self.last_retrain_count
        if new_trades < self.retrain_threshold:
            return {
                "retrained": False,
                "reason": "threshold_not_met",
                "total_trades": total_trades,
                "new_since_last": new_trades,
                "threshold": self.retrain_threshold,
                "remaining": self.retrain_threshold - new_trades,
            }

        return self._retrain_all(pipeline_df, outcomes_df)

    def _retrain_all(self, pipeline_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> dict:
        """Retrain all intelligence components."""
        results = {}
        success = True

        if pipeline_df is not None and not pipeline_df.empty:
            try:
                regime_result = self.regime_detector.train(pipeline_df, retrain=True)
                results["regime_detector"] = regime_result
                logger.info("Regime detector retrained on %d samples", regime_result.get("samples", 0))
            except Exception as e:
                logger.error("Regime detector retrain failed: %s", e)
                results["regime_detector"] = {"error": str(e)}
                success = False
        else:
            results["regime_detector"] = {"skipped": "no_pipeline_data"}

        if not outcomes_df.empty:
            try:
                new_outcomes = outcomes_df.iloc[self.last_retrain_count :].copy()
                learned = 0
                for _, outcome in new_outcomes.iterrows():
                    self.rl_agent.learn_from_outcome(outcome.to_dict())
                    learned += 1
                if hasattr(self.rl_agent, "_save"):
                    self.rl_agent._save()

                rl_stats = self.rl_agent.get_stats()
                results["rl_agent"] = {
                    "retrained": True,
                    "outcomes_learned": learned,
                    "total_trades_seen": rl_stats["total_trades_seen"],
                    "win_rate": rl_stats["win_rate_pct"],
                    "epsilon": rl_stats["epsilon"],
                }
                logger.info(
                    "RL Agent learned %d new outcomes: %d trades, %.1f%% win rate",
                    learned,
                    rl_stats["total_trades_seen"],
                    rl_stats["win_rate_pct"],
                )
            except Exception as e:
                logger.error("RL Agent update failed: %s", e)
                results["rl_agent"] = {"error": str(e)}
                success = False

        try:
            from smart_monitor.simple_learner import SimpleTradeLearner

            learner = SimpleTradeLearner()
            results["simple_trade_learner"] = {"trained": learner.train(force_retrain=True)}
        except Exception as e:
            logger.error("SimpleTradeLearner retrain failed: %s", e)
            results["simple_trade_learner"] = {"error": str(e)}
            success = False

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
                outcomes_df = pd.read_csv(self.outcomes_path, parse_dates=["entry_time"], on_bad_lines="skip")
            except Exception:
                outcomes_df = pd.DataFrame()

        original_count = self.last_retrain_count
        self.last_retrain_count = 0
        result = self._retrain_all(pipeline_df, outcomes_df)
        if not result.get("retrained"):
            self.last_retrain_count = original_count
        return result

    def _save_marker(self):
        """Save retrain state to disk."""
        try:
            self.retrain_marker_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.retrain_marker_path, "w", encoding="utf-8") as f:
                f.write(f"{self.last_retrain_count}\n")
                f.write(f"{self.last_retrain_time.isoformat() if self.last_retrain_time else 'never'}\n")
                f.write(f"{self.total_retrains}\n")
        except Exception as e:
            logger.warning("Could not save retrain marker: %s", e)

    def _load_marker(self):
        """Load retrain state from disk."""
        try:
            if self.retrain_marker_path.exists():
                with open(self.retrain_marker_path, "r", encoding="utf-8") as f:
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
            logger.warning("Could not load retrain marker: %s", e)

    def get_status(self) -> dict:
        """Get retraining pipeline status."""
        total = 0
        if self.outcomes_path.exists():
            try:
                total = len(pd.read_csv(self.outcomes_path, on_bad_lines="skip"))
            except Exception:
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
