"""
Trade Performance Tracker - Monitors and analyzes trade outcomes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from core.logging_utils import get_logger

logger = get_logger(__name__)


class TradePerformanceTracker:
    """Tracks and analyzes trade performance to identify patterns"""
    
    def __init__(self, audit_path="data/live/execution_audit.csv"):
        self.audit_path = Path(audit_path)
        self.lookback_trades = 30  # Analyze last 30 trades
        self.min_trades_for_analysis = 10
        
    def get_performance_summary(self, system="FLOW_EXP", lookback_trades=None):
        """Get comprehensive performance summary"""
        if not self.audit_path.exists():
            return self._empty_summary()
        
        try:
            df = pd.read_csv(self.audit_path, parse_dates=['time', 'signal_time'])
            
            # Filter by system
            if system:
                df = df[df['system'] == system]
            
            if df.empty:
                return self._empty_summary()
            
            # Get recent trades
            if lookback_trades:
                df = df.tail(lookback_trades)
            else:
                df = df.tail(self.lookback_trades)
            
            # Only analyze executed trades
            executed = df[df['status'] == 'EXECUTED']
            
            if executed.empty:
                return self._empty_summary()
            
            # Calculate metrics
            total_trades = len(executed)
            avg_spread = executed['spread'].mean()
            avg_lot = executed['lot'].mean()
            
            # Analyze by hour
            executed['hour'] = pd.to_datetime(executed['time']).dt.hour
            hour_dist = executed.groupby('hour').size().to_dict()
            
            # Analyze by regime
            regime_dist = executed['regime'].value_counts().to_dict()
            
            # Analyze by setup
            setup_dist = executed['setup'].value_counts().to_dict()
            
            # Analyze by quality
            quality_dist = executed['comment'].str.extract(r'_(ELITE|HIGH|MEDIUM|LOW)').dropna()[0].value_counts().to_dict()
            
            return {
                'total_trades': total_trades,
                'avg_spread': avg_spread,
                'avg_lot': avg_lot,
                'hour_distribution': hour_dist,
                'regime_distribution': regime_dist,
                'setup_distribution': setup_dist,
                'quality_distribution': quality_dist,
                'time_range': {
                    'start': df['time'].min(),
                    'end': df['time'].max()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return self._empty_summary()
    
    def identify_problem_patterns(self, system="FLOW_EXP"):
        """Identify patterns that lead to poor performance"""
        if not self.audit_path.exists():
            return []
        
        try:
            df = pd.read_csv(self.audit_path, parse_dates=['time', 'signal_time'])
            
            # Focus on executed trades
            executed = df[df['status'] == 'EXECUTED']
            if executed.empty:
                return []
            
            problems = []
            
            # 1. Identify hours with high activity but likely poor results
            executed['hour'] = pd.to_datetime(executed['time']).dt.hour
            hour_counts = executed.groupby('hour').size()
            
            # Flag hours with >20% of total trades
            total = len(executed)
            for hour, count in hour_counts.items():
                if count / total > 0.20:
                    problems.append({
                        'type': 'high_activity_hour',
                        'value': hour,
                        'count': count,
                        'percentage': count / total * 100,
                        'recommendation': f"Consider reducing trades during hour {hour}"
                    })
            
            # 2. Identify regimes with high activity
            regime_counts = executed['regime'].value_counts()
            for regime, count in regime_counts.items():
                if count / total > 0.35:  # More than 35% of trades in one regime
                    problems.append({
                        'type': 'regime_concentration',
                        'value': regime,
                        'count': count,
                        'percentage': count / total * 100,
                        'recommendation': f"High concentration in {regime} regime - ensure proper filtering"
                    })
            
            # 3. Identify spread issues
            high_spread_trades = executed[executed['spread'] > 20]
            if len(high_spread_trades) / total > 0.30:
                problems.append({
                    'type': 'high_spread_prevalence',
                    'value': len(high_spread_trades),
                    'percentage': len(high_spread_trades) / total * 100,
                    'recommendation': "Many trades with high spread - tighten spread filter"
                })
            
            # 4. Identify low quality trades
            if 'quality_distribution' in self.get_performance_summary(system):
                quality_dist = self.get_performance_summary(system)['quality_distribution']
                medium_plus = quality_dist.get('MEDIUM', 0) + quality_dist.get('LOW', 0)
                if medium_plus / total > 0.50:
                    problems.append({
                        'type': 'low_quality_prevalence',
                        'value': medium_plus,
                        'percentage': medium_plus / total * 100,
                        'recommendation': "Too many MEDIUM/LOW quality trades - raise minimum score"
                    })
            
            return problems
            
        except Exception as e:
            logger.error(f"Error identifying problem patterns: {e}")
            return []
    
    def get_trading_patterns(self, system="FLOW_EXP"):
        """Analyze successful vs unsuccessful patterns"""
        if not self.audit_path.exists():
            return None
        
        try:
            df = pd.read_csv(self.audit_path, parse_dates=['time', 'signal_time'])
            executed = df[df['status'] == 'EXECUTED']
            
            if executed.empty:
                return None
            
            # Group by key characteristics
            patterns = {}
            
            # By hour
            executed['hour'] = pd.to_datetime(executed['time']).dt.hour
            patterns['by_hour'] = executed.groupby('hour').agg({
                'spread': 'mean',
                'lot': 'mean',
                'signal_time': 'count'
            }).to_dict()
            
            # By regime
            patterns['by_regime'] = executed.groupby('regime').agg({
                'spread': 'mean',
                'lot': 'mean',
                'signal_time': 'count'
            }).to_dict()
            
            # By setup
            patterns['by_setup'] = executed.groupby('setup').agg({
                'spread': 'mean',
                'lot': 'mean',
                'signal_time': 'count'
            }).to_dict()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting trading patterns: {e}")
            return None
    
    def _empty_summary(self):
        """Return empty summary structure"""
        return {
            'total_trades': 0,
            'avg_spread': 0,
            'avg_lot': 0,
            'hour_distribution': {},
            'regime_distribution': {},
            'setup_distribution': {},
            'quality_distribution': {},
            'time_range': {'start': None, 'end': None}
        }