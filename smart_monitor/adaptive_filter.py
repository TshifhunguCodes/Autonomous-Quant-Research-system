"""
Adaptive Filter - Dynamically adjusts filtering based on performance
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date
from core.logging_utils import get_logger

logger = get_logger(__name__)


class AdaptiveFilter:
    """Dynamically adjusts trade filters based on recent performance"""
    
    def __init__(self, config_path="config/app_config.json"):
        self.config_path = Path(config_path)
        self.audit_path = Path("data/live/execution_audit.csv")
        self.performance_path = Path("data/live/smart_performance.csv")
        
        # Default thresholds
        self.base_thresholds = {
            'min_quality_score': 60,
            'max_spread': 25,
            'min_confirm_score': 55,
            'min_confluence_count': 2,
            'bad_hours': [2, 3, 4, 5, 14, 15],  # Hours to potentially block
            'max_trades_per_hour': 3,
            'choppy_regime_multiplier': 0.5,
        }
        
        # Current active thresholds (may be adjusted)
        self.active_thresholds = self.base_thresholds.copy()
        
        # Performance tracking
        self.daily_stats = {}
        
    def get_adaptive_filters(self, signal):
        """
        Get dynamically adjusted filters for current signal
        
        Returns:
            tuple: (allow_trade: bool, reason: str, adjusted_lot_multiplier: float)
        """
        # Update thresholds based on recent performance
        self._update_thresholds()
        
        # Get signal characteristics
        current_hour = pd.to_datetime(signal['time']).hour
        current_regime = signal.get('market_regime', 'UNKNOWN')
        current_spread = float(signal.get('spread', 0))
        quality_score = float(signal.get('smart_quality_score', 60))
        
        # 1. Check hour-based filter
        if current_hour in self.active_thresholds['bad_hours']:
            # Only allow high quality trades during bad hours
            if quality_score < 75:
                return False, f"ADAPTIVE_HOUR_BLOCK (hour {current_hour}, quality {quality_score:.0f})", 0
        
        # 2. Check spread filter
        if current_spread > self.active_thresholds['max_spread']:
            return False, f"ADAPTIVE_SPREAD_BLOCK (spread {current_spread:.1f} > {self.active_thresholds['max_spread']})", 0
        
        # 3. Check quality score
        if quality_score < self.active_thresholds['min_quality_score']:
            return False, f"ADAPTIVE_QUALITY_BLOCK (score {quality_score:.0f} < {self.active_thresholds['min_quality_score']})", 0
        
        # 4. Check regime-specific filters
        lot_multiplier = 1.0
        if current_regime == 'CHOPPY':
            lot_multiplier *= self.active_thresholds['choppy_regime_multiplier']
            if quality_score < 70:
                return False, "ADAPTIVE_CHOPPY_BLOCK (low quality in choppy)", 0
        
        # 5. Check trade frequency
        if self._is_too_active_current_hour():
            if quality_score < 75:
                return False, "ADAPTIVE_FREQUENCY_BLOCK (too many trades this hour)", 0
        
        # 6. Apply dynamic lot sizing based on quality
        if quality_score >= 80:
            lot_multiplier *= 1.2  # High quality - increase size
        elif quality_score >= 70:
            lot_multiplier *= 1.0  # Normal size
        elif quality_score >= 60:
            lot_multiplier *= 0.7  # Reduce size
        else:
            lot_multiplier *= 0.3  # Minimal size
        
        return True, "ADAPTIVE_PASS", lot_multiplier
    
    def _update_thresholds(self):
        """Update thresholds based on recent performance"""
        if not self.audit_path.exists():
            return
        
        try:
            df = pd.read_csv(self.audit_path, parse_dates=['time', 'signal_time'])
            flow_df = df[df['system'] == 'FLOW_EXP']
            
            if len(flow_df) < 10:
                return  # Not enough data
            
            # Get recent trades (last 20)
            recent = flow_df.tail(20)
            executed = recent[recent['status'] == 'EXECUTED']
            
            if len(executed) < 5:
                return
            
            # Analyze spread distribution
            avg_spread = executed['spread'].mean()
            if avg_spread > 18:
                # Spreads are generally high - relax spread filter
                self.active_thresholds['max_spread'] = min(30, avg_spread * 1.5)
            else:
                self.active_thresholds['max_spread'] = self.base_thresholds['max_spread']
            
            # Analyze quality score distribution
            # Extract quality from comment if available
            quality_scores = self._extract_quality_from_comments(executed)
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                if avg_quality < 60:
                    # Quality is generally low - raise minimum
                    self.active_thresholds['min_quality_score'] = 65
                else:
                    self.active_thresholds['min_quality_score'] = self.base_thresholds['min_quality_score']
            
            # Analyze hourly distribution
            executed['hour'] = pd.to_datetime(executed['time']).dt.hour
            hour_counts = executed.groupby('hour').size()
            
            # Identify problematic hours (>30% of trades)
            total = len(executed)
            bad_hours = []
            for hour, count in hour_counts.items():
                if count / total > 0.30:
                    bad_hours.append(hour)
            
            if bad_hours:
                self.active_thresholds['bad_hours'] = bad_hours
            
        except Exception as e:
            logger.error(f"Error updating adaptive thresholds: {e}")
    
    def _extract_quality_from_comments(self, df):
        """Extract quality scores from trade comments"""
        scores = []
        for comment in df['comment']:
            if pd.isna(comment):
                continue
            comment = str(comment)
            if 'ELITE' in comment:
                scores.append(85)
            elif 'HIGH' in comment:
                scores.append(75)
            elif 'MEDIUM' in comment:
                scores.append(55)
            elif 'LOW' in comment:
                scores.append(35)
        return scores
    
    def _is_too_active_current_hour(self):
        """Check if we've exceeded trade limit for current hour"""
        if not self.audit_path.exists():
            return False
        
        try:
            df = pd.read_csv(self.audit_path, parse_dates=['time', 'signal_time'])
            today = date.today()
            
            # Filter for today's trades
            df['trade_date'] = df['time'].dt.date
            today_trades = df[df['trade_date'] == today]
            
            if today_trades.empty:
                return False
            
            # Count trades in current hour
            current_hour = datetime.now().hour
            today_trades['hour'] = pd.to_datetime(today_trades['time']).dt.hour
            hour_trades = today_trades[today_trades['hour'] == current_hour]
            
            return len(hour_trades) >= self.active_thresholds['max_trades_per_hour']
            
        except Exception as e:
            logger.error(f"Error checking hourly activity: {e}")
            return False
    
    def get_recommendations(self):
        """Get recommendations based on current performance"""
        recommendations = []
        
        # Check if thresholds have been adjusted
        if self.active_thresholds['max_spread'] != self.base_thresholds['max_spread']:
            recommendations.append(
                f"Spread threshold adjusted to {self.active_thresholds['max_spread']:.0f} "
                f"(base: {self.base_thresholds['max_spread']})"
            )
        
        if self.active_thresholds['min_quality_score'] != self.base_thresholds['min_quality_score']:
            recommendations.append(
                f"Minimum quality score adjusted to {self.active_thresholds['min_quality_score']} "
                f"(base: {self.base_thresholds['min_quality_score']})"
            )
        
        if self.active_thresholds['bad_hours'] != self.base_thresholds['bad_hours']:
            recommendations.append(
                f"Restricted hours updated to {self.active_thresholds['bad_hours']}"
            )
        
        return recommendations
    
    def reset_to_base(self):
        """Reset all thresholds to base values"""
        self.active_thresholds = self.base_thresholds.copy()
        logger.info("Adaptive filters reset to base thresholds")