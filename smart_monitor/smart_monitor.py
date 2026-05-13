"""
Smart Monitor - Main integration module for intelligent trade filtering
Combines performance tracking, quality scoring, adaptive filtering, and ML learning
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from core.logging_utils import get_logger

from smart_monitor.performance_tracker import TradePerformanceTracker
from smart_monitor.quality_scorer import TradeQualityScorer
from smart_monitor.adaptive_filter import AdaptiveFilter
from smart_monitor.simple_learner import SimpleTradeLearner

logger = get_logger(__name__)


class SmartMonitor:
    """
    Central smart monitoring system that combines all intelligent filtering components
    """
    
    def __init__(self):
        self.performance_tracker = TradePerformanceTracker()
        self.quality_scorer = TradeQualityScorer()
        self.adaptive_filter = AdaptiveFilter()
        self.ml_learner = SimpleTradeLearner()
        
        # Try to load existing ML model
        self.ml_learner.load_model()
        
        # Statistics
        self.total_signals = 0
        self.allowed_signals = 0
        self.blocked_signals = 0
        self.block_reasons = {}
    
    def evaluate_signal(self, signal, system="FLOW_EXP"):
        """
        Comprehensive signal evaluation using all smart monitoring components
        
        Args:
            signal: dict containing signal data
            system: "FLOW_EXP" or "ALPHA"
            
        Returns:
            tuple: (allow: bool, quality_score: float, lot_multiplier: float, reason: str)
        """
        self.total_signals += 1
        
        # 1. Calculate comprehensive quality score
        quality_score = self.quality_scorer.score_trade(signal)
        signal['smart_quality_score'] = quality_score
        quality_tier = self.quality_scorer.get_quality_tier(quality_score)
        
        # 2. Get ML prediction
        ml_allow, ml_probability, ml_reason = self.ml_learner.should_allow_trade(signal)
        signal['ml_success_probability'] = ml_probability
        
        # 3. Get adaptive filter decision
        adaptive_allow, adaptive_reason, adaptive_lot_mult = self.adaptive_filter.get_adaptive_filters(signal)
        
        # 4. Combine all decisions
        allow = True
        reasons = []
        lot_multiplier = 1.0
        
        # Quality-based decision
        if system == "FLOW_EXP":
            # Stricter requirements for FLOW
            if quality_score < 55:
                allow = False
                reasons.append(f"LOW_QUALITY_SCORE ({quality_score:.0f})")
            elif quality_score < 65:
                lot_multiplier *= 0.5  # Significantly reduce size for marginal quality
            elif quality_score < 75:
                lot_multiplier *= 0.8  # Slightly reduce size
        else:
            # ALPHA requirements
            if quality_score < 65:
                allow = False
                reasons.append(f"LOW_QUALITY_SCORE ({quality_score:.0f})")
        
        # ML-based decision (only for FLOW, ALPHA uses stricter rules)
        if system == "FLOW_EXP" and not ml_allow:
            allow = False
            reasons.append(ml_reason)
        
        # Adaptive filter decision
        if not adaptive_allow:
            allow = False
            reasons.append(adaptive_reason)
        
        # Combine lot multipliers
        lot_multiplier *= adaptive_lot_mult
        
        # Apply quality-based lot adjustment
        if quality_score >= 80:
            lot_multiplier *= 1.2
        elif quality_score >= 70:
            lot_multiplier *= 1.0
        elif quality_score >= 60:
            lot_multiplier *= 0.7
        else:
            lot_multiplier *= 0.4
        
        # Final decision
        if allow:
            self.allowed_signals += 1
            reason = f"SMART_APPROVE (quality={quality_score:.0f}, ml_p={ml_probability:.2f})"
        else:
            self.blocked_signals += 1
            reason = "; ".join(reasons)
            self._track_block_reason(reasons)
        
        # Log statistics periodically
        if self.total_signals % 10 == 0:
            self._log_statistics()
        
        return allow, quality_score, lot_multiplier, reason, quality_tier
    
    def train_ml_model(self):
        """Train the ML model on recent data"""
        success = self.ml_learner.train()
        if success:
            logger.info("Smart Monitor: ML model retrained successfully")
        return success
    
    def get_performance_report(self):
        """Get comprehensive performance report"""
        report = {
            'summary': {
                'total_signals': self.total_signals,
                'allowed_signals': self.allowed_signals,
                'blocked_signals': self.blocked_signals,
                'allow_rate': self.allowed_signals / max(self.total_signals, 1) * 100,
            },
            'quality_distribution': self._get_quality_distribution(),
            'block_reasons': dict(self.block_reasons),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'problem_patterns': self.performance_tracker.identify_problem_patterns(),
            'adaptive_recommendations': self.adaptive_filter.get_recommendations(),
        }
        return report
    
    def _get_quality_distribution(self):
        """Get distribution of quality scores"""
        # This would be tracked over time
        return {
            'elite': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
    
    def _track_block_reason(self, reasons):
        """Track reasons for blocking signals"""
        for reason in reasons:
            # Extract main reason category
            category = reason.split()[0] if reason else "UNKNOWN"
            self.block_reasons[category] = self.block_reasons.get(category, 0) + 1
    
    def _log_statistics(self):
        """Log current statistics"""
        logger.info(
            f"Smart Monitor Stats: Total={self.total_signals}, "
            f"Allowed={self.allowed_signals}, Blocked={self.blocked_signals}, "
            f"Allow Rate={self.allowed_signals/max(self.total_signals,1)*100:.1f}%"
        )
    
    def reset_statistics(self):
        """Reset monitoring statistics"""
        self.total_signals = 0
        self.allowed_signals = 0
        self.blocked_signals = 0
        self.block_reasons = {}


# Singleton instance
_smart_monitor_instance = None


def get_smart_monitor():
    """Get the singleton SmartMonitor instance"""
    global _smart_monitor_instance
    if _smart_monitor_instance is None:
        _smart_monitor_instance = SmartMonitor()
    return _smart_monitor_instance