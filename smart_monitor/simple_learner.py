"""
Simple Trade Learner - Lightweight ML model for trade filtering
Uses a simple scoring-based learning approach that works well with limited data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from core.logging_utils import get_logger

logger = get_logger(__name__)


class SimpleTradeLearner:
    """
    Simple machine learning model that learns from trade outcomes
    Uses feature importance weighting to predict trade success
    """
    
    def __init__(self, model_path="data/ai/simple_model.json"):
        self.model_path = Path(model_path)
        self.training_data_path = Path("data/ai/training_data.csv")
        self.min_trades_for_learning = 15
        self.feature_weights = self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize feature weights"""
        return {
            'confluence_count': 0.20,
            'htf_alignment': 0.20,
            'momentum_score': 0.15,
            'volume_ratio': 0.10,
            'spread_quality': 0.10,
            'confirm_score': 0.10,
            'structure_score': 0.08,
            'hour_quality': 0.04,
            'regime_suitability': 0.03
        }
    
    def train(self, force_retrain=False):
        """Train the model on historical trade data"""
        audit_path = Path("data/live/execution_audit.csv")
        if not audit_path.exists():
            return False
        
        try:
            df = pd.read_csv(audit_path, parse_dates=['time', 'signal_time'])
            flow_df = df[df['system'] == 'FLOW_EXP']
            
            if len(flow_df) < self.min_trades_for_learning:
                logger.info(f"Not enough trades for learning ({len(flow_df)} < {self.min_trades_for_learning})")
                return False
            
            # Extract features and outcomes
            features, outcomes = self._extract_features_and_outcomes(flow_df)
            
            if len(outcomes) < self.min_trades_for_learning:
                return False
            
            # Calculate feature importance based on correlation with outcomes
            self._update_feature_weights(features, outcomes)
            
            # Save model
            self._save_model()
            
            logger.info(f"SimpleTradeLearner trained on {len(outcomes)} trades")
            return True
            
        except Exception as e:
            logger.error(f"Error training SimpleTradeLearner: {e}")
            return False
    
    def _extract_features_and_outcomes(self, df):
        """Extract features and outcomes from trade data"""
        features = []
        outcomes = []
        
        for _, row in df.iterrows():
            if row['status'] != 'EXECUTED':
                continue
            
            # Extract features
            feature_vector = {
                'confluence_count': self._count_confluences(row),
                'htf_alignment': 1 if row.get('htf_bias', 'NEUTRAL') != 'NEUTRAL' else 0,
                'momentum_score': self._calculate_momentum(row),
                'volume_ratio': float(row.get('volume', 1) or 1) / max(float(row.get('volume_avg_20', 1) or 1), 0.1),
                'spread_quality': max(0, 100 - float(row.get('spread', 0))) / 100,
                'confirm_score': float(row.get('confirm_score', 50)) / 100,
                'structure_score': self._calculate_structure(row),
                'hour_quality': self._get_hour_quality(pd.to_datetime(row['time']).hour),
                'regime_suitability': self._get_regime_suitability(row.get('regime', 'UNKNOWN'))
            }
            
            features.append(feature_vector)
            
            # Estimate outcome (simplified - would be better with actual outcomes)
            # Use spread and quality as proxy for expected outcome
            quality_score = self._estimate_quality(row)
            outcome = 1 if quality_score > 65 else 0  # Binary outcome
            outcomes.append(outcome)
        
        return features, outcomes
    
    def _count_confluences(self, row):
        """Count technical confluences"""
        count = 0
        if row.get('fvg_bullish') or row.get('fvg_bearish'):
            count += 1
        if row.get('order_block') or row.get('demand_zone') or row.get('supply_zone'):
            count += 1
        if row.get('liquidity_sweep') or row.get('sweep_high') or row.get('sweep_low'):
            count += 1
        if row.get('bos') or row.get('choch'):
            count += 1
        if row.get('major_support') or row.get('major_resistance'):
            count += 1
        return min(count / 5, 1.0)  # Normalize to 0-1
    
    def _calculate_momentum(self, row):
        """Calculate momentum score"""
        rsi = float(row.get('rsi14', 50) or 50)
        direction = str(row.get('side', '')).upper()
        
        if direction == 'BUY':
            if 50 < rsi < 70:
                return 1.0
            elif rsi <= 50 or rsi >= 70:
                return 0.3
        else:  # SELL
            if 30 < rsi < 50:
                return 1.0
            elif rsi >= 50 or rsi <= 30:
                return 0.3
        return 0.5
    
    def _calculate_structure(self, row):
        """Calculate structure score"""
        score = 0.5
        if row.get('bos'):
            score += 0.2
        if row.get('choch'):
            score += 0.2
        lifecycle = row.get('lifecycle_state', 'TREND_HEALTHY')
        if lifecycle in ['TREND_HEALTHY', 'BREAKOUT_EXPANSION']:
            score += 0.1
        elif lifecycle in ['TREND_EXHAUSTING', 'EXIT_WARNING']:
            score -= 0.2
        return max(0, min(1, score))
    
    def _get_hour_quality(self, hour):
        """Get hour quality score (0-1)"""
        # Good trading hours
        good_hours = [7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21]
        return 1.0 if hour in good_hours else 0.5
    
    def _get_regime_suitability(self, regime):
        """Get regime suitability score (0-1)"""
        regime_scores = {
            'TREND_UP': 0.9,
            'TREND_DOWN': 0.9,
            'BREAKOUT': 0.8,
            'RANGE': 0.6,
            'REVERSAL': 0.5,
            'CHOPPY': 0.3,
            'VOLATILE': 0.4
        }
        return regime_scores.get(regime, 0.5)
    
    def _estimate_quality(self, row):
        """Estimate trade quality from available data"""
        score = 50
        
        # Base score
        score += float(row.get('confirm_score', 50) or 50) * 0.3
        
        # Quality from comment
        comment = str(row.get('comment', ''))
        if 'ELITE' in comment:
            score += 20
        elif 'HIGH' in comment:
            score += 10
        elif 'MEDIUM' in comment:
            score += 0
        elif 'LOW' in comment:
            score -= 10
        
        # Spread penalty
        spread = float(row.get('spread', 0) or 0)
        if spread > 20:
            score -= 15
        elif spread > 15:
            score -= 8
        
        return score
    
    def _update_feature_weights(self, features, outcomes):
        """Update feature weights based on correlation with outcomes"""
        if not features or not outcomes:
            return
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(features)
        df['outcome'] = outcomes
        
        # Calculate correlation for each feature
        correlations = {}
        for feature in self.feature_weights.keys():
            if feature in df.columns:
                corr = df[feature].corr(df['outcome'])
                # Use absolute correlation as importance
                correlations[feature] = abs(corr) if not pd.isna(corr) else 0.1
        
        # Normalize weights
        total = sum(correlations.values())
        if total > 0:
            for feature in self.feature_weights:
                self.feature_weights[feature] = correlations.get(feature, 0.1) / total
    
    def predict_success_probability(self, signal):
        """
        Predict probability of trade success
        
        Returns:
            float: Success probability 0-1
        """
        # Extract features from signal
        features = {
            'confluence_count': self._count_confluences(signal),
            'htf_alignment': 1 if signal.get('htf_bias', 'NEUTRAL') != 'NEUTRAL' else 0,
            'momentum_score': self._calculate_momentum(signal),
            'volume_ratio': float(signal.get('volume', 1) or 1) / max(float(signal.get('volume_avg_20', 1) or 1), 0.1),
            'spread_quality': max(0, 100 - float(signal.get('spread', 0))) / 100,
            'confirm_score': float(signal.get('confirm_score', 50)) / 100,
            'structure_score': self._calculate_structure(signal),
            'hour_quality': self._get_hour_quality(pd.to_datetime(signal['time']).hour),
            'regime_suitability': self._get_regime_suitability(signal.get('market_regime', 'UNKNOWN'))
        }
        
        # Calculate weighted score
        weighted_score = 0
        for feature, weight in self.feature_weights.items():
            weighted_score += features.get(feature, 0.5) * weight
        
        # Convert to probability (sigmoid-like transformation)
        probability = 1 / (1 + np.exp(-10 * (weighted_score - 0.5)))
        
        return probability
    
    def should_allow_trade(self, signal, threshold=0.5):
        """
        Determine if trade should be allowed based on ML prediction
        
        Args:
            signal: Trade signal data
            threshold: Minimum probability threshold (default 0.5)
            
        Returns:
            tuple: (allow: bool, probability: float, reason: str)
        """
        probability = self.predict_success_probability(signal)
        
        if probability >= threshold:
            return True, probability, f"ML_APPROVE (p={probability:.2f})"
        else:
            return False, probability, f"ML_REJECT (p={probability:.2f} < {threshold})"
    
    def _save_model(self):
        """Save model to file"""
        model_data = {
            'feature_weights': self.feature_weights,
            'last_trained': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self):
        """Load model from file"""
        if not self.model_path.exists():
            return False
        
        try:
            import json
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            
            self.feature_weights = model_data.get('feature_weights', self.feature_weights)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False