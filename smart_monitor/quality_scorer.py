"""
Trade Quality Scorer - Evaluates trade quality before entry
"""

import numpy as np
import pandas as pd
from core.logging_utils import get_logger

logger = get_logger(__name__)


class TradeQualityScorer:
    """Scores trade quality on a 0-100 scale"""
    
    def __init__(self):
        # Weights for different quality factors
        self.weights = {
            'confluence': 0.25,      # Technical confluences
            'trend_alignment': 0.20, # HTF alignment
            'momentum': 0.15,        # Momentum indicators
            'volume': 0.10,          # Volume confirmation
            'spread': 0.10,          # Spread quality
            'score': 0.10,           # Base confirmation score
            'structure': 0.10        # Market structure
        }
    
    def score_trade(self, signal):
        """
        Calculate comprehensive quality score (0-100)
        
        Args:
            signal: dict containing signal data
            
        Returns:
            float: Quality score 0-100
        """
        # 1. Confluence Score (0-100)
        confluence_score = self._calculate_confluence_score(signal)
        
        # 2. Trend Alignment Score (0-100)
        trend_score = self._calculate_trend_alignment_score(signal)
        
        # 3. Momentum Score (0-100)
        momentum_score = self._calculate_momentum_score(signal)
        
        # 4. Volume Score (0-100)
        volume_score = self._calculate_volume_score(signal)
        
        # 5. Spread Score (0-100)
        spread_score = self._calculate_spread_score(signal)
        
        # 6. Base Score (0-100)
        # Support both confirm_score and flow_score for backtest compatibility
        base_score = float(signal.get('confirm_score', signal.get('flow_score', 50)))
        
        # 7. Structure Score (0-100)
        structure_score = self._calculate_structure_score(signal)
        
        # Calculate weighted total
        total_score = (
            confluence_score * self.weights['confluence'] +
            trend_score * self.weights['trend_alignment'] +
            momentum_score * self.weights['momentum'] +
            volume_score * self.weights['volume'] +
            spread_score * self.weights['spread'] +
            base_score * self.weights['score'] +
            structure_score * self.weights['structure']
        )
        
        return max(0, min(100, total_score))
    
    def _calculate_confluence_score(self, signal):
        """Calculate score based on technical confluences (0-100)"""
        confluences = 0
        max_confluences = 6
        
        # FVG (Fair Value Gap) - support multiple field names
        if self._truthy(signal.get('fvg_bullish')) or self._truthy(signal.get('fvg_bearish')) or self._truthy(signal.get('fvg_zone')):
            confluences += 1
        
        # Order Block - support multiple field names
        if self._truthy(signal.get('order_block')) or self._truthy(signal.get('demand_zone')) or self._truthy(signal.get('supply_zone')):
            confluences += 1
        
        # Liquidity Sweep
        if self._truthy(signal.get('liquidity_sweep')) or self._truthy(signal.get('sweep_high')) or self._truthy(signal.get('sweep_low')):
            confluences += 1
        
        # HTF Bias Alignment
        htf_bias = self._normalize_bias(signal.get('htf_bias', signal.get('h1_bias', 'NEUTRAL')))
        if htf_bias != 'NEUTRAL':
            confluences += 1
        
        # Major Support/Resistance
        if self._truthy(signal.get('major_support')) or self._truthy(signal.get('major_resistance')) or self._truthy(signal.get('support_level')) or self._truthy(signal.get('resistance_level')):
            confluences += 1
        
        # BOS/CHOCH - support multiple field names
        if self._truthy(signal.get('bos')) or self._truthy(signal.get('bos_up')) or self._truthy(signal.get('bos_down')) or self._truthy(signal.get('choch')):
            confluences += 1
        
        return (confluences / max_confluences) * 100
    
    def _calculate_trend_alignment_score(self, signal):
        """Calculate score based on trend alignment (0-100)"""
        # Support both confirmed_signal and direction for backtest compatibility
        direction = str(signal.get('confirmed_signal', signal.get('direction', ''))).upper()
        # Convert LONG/SHORT to BUY/SELL
        if direction == 'LONG':
            direction = 'BUY'
        elif direction == 'SHORT':
            direction = 'SELL'
        htf_bias = self._normalize_bias(signal.get('htf_bias', signal.get('h1_bias', 'NEUTRAL')))
        # Support multiple field names for market state
        market_state = str(signal.get('market_state', signal.get('market_regime', signal.get('behavior_label', 'UNKNOWN')))).upper()
        
        score = 50  # Neutral starting point
        
        # HTF Alignment
        if htf_bias != 'NEUTRAL':
            if (direction == 'BUY' and htf_bias == 'BULLISH') or \
               (direction == 'SELL' and htf_bias == 'BEARISH'):
                score += 30  # Strong alignment
            else:
                score -= 30  # Misalignment
        else:
            score -= 10  # No HTF bias
        
        # Market State Alignment - support various state names
        if market_state in ['TREND_UP', 'TREND_DOWN', 'UPTREND', 'DOWNTREND']:
            # Normalize
            if market_state in ['UPTREND']:
                market_state = 'TREND_UP'
            elif market_state in ['DOWNTREND']:
                market_state = 'TREND_DOWN'
            if (direction == 'BUY' and market_state == 'TREND_UP') or \
               (direction == 'SELL' and market_state == 'TREND_DOWN'):
                score += 20  # Trading with trend
            else:
                score -= 20  # Counter-trend
        
        return max(0, min(100, score))

    def _normalize_bias(self, value):
        bias = str(value or 'NEUTRAL').strip().upper()
        aliases = {
            'LONG': 'BULLISH',
            'UP': 'BULLISH',
            'BUY': 'BULLISH',
            'SHORT': 'BEARISH',
            'DOWN': 'BEARISH',
            'SELL': 'BEARISH',
        }
        return aliases.get(bias, bias if bias in {'BULLISH', 'BEARISH', 'NEUTRAL'} else 'NEUTRAL')

    def _truthy(self, value):
        if value is None:
            return False
        try:
            if np.isscalar(value) and pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y'}
        return bool(value)
    
    def _calculate_momentum_score(self, signal):
        """Calculate score based on momentum indicators (0-100)"""
        # Support both confirmed_signal and direction for backtest compatibility
        direction = str(signal.get('confirmed_signal', signal.get('direction', ''))).upper()
        # Convert LONG/SHORT to BUY/SELL
        if direction == 'LONG':
            direction = 'BUY'
        elif direction == 'SHORT':
            direction = 'SELL'
        rsi = float(signal.get('rsi14', 50))
        
        score = 50  # Neutral
        
        if direction == 'BUY':
            # For buys, want RSI > 50 but not overbought
            if 50 < rsi < 70:
                score += 25
            elif rsi <= 50:
                score -= 15
            elif rsi >= 70:
                score -= 10  # Overbought
        else:  # SELL
            # For sells, want RSI < 50 but not oversold
            if 30 < rsi < 50:
                score += 25
            elif rsi >= 50:
                score -= 15
            elif rsi <= 30:
                score -= 10  # Oversold
        
        # Continuation strength
        continuation = float(signal.get('continuation_strength', 50))
        if continuation > 70:
            score += 15
        elif continuation > 50:
            score += 5
        elif continuation < 30:
            score -= 15
        
        return max(0, min(100, score))
    
    def _calculate_volume_score(self, signal):
        """Calculate score based on volume confirmation (0-100)"""
        # Support various volume field names for backtest compatibility
        volume_avg = float(signal.get('volume_avg_20', signal.get('avg_tr_20', 1.0)) or 1.0)
        current_volume = float(signal.get('volume', signal.get('tick_volume', 1.0)) or 1.0)
        
        if volume_avg <= 0:
            return 50
        
        volume_ratio = current_volume / volume_avg
        
        if volume_ratio > 1.5:
            return 90  # Strong volume
        elif volume_ratio > 1.2:
            return 75  # Good volume
        elif volume_ratio > 0.8:
            return 60  # Average volume
        elif volume_ratio > 0.5:
            return 40  # Low volume
        else:
            return 20  # Very low volume
    
    def _calculate_spread_score(self, signal):
        """Calculate score based on spread quality (0-100)"""
        spread = float(signal.get('spread', 0))
        
        if spread <= 5:
            return 100  # Excellent
        elif spread <= 10:
            return 85   # Good
        elif spread <= 15:
            return 70   # Acceptable
        elif spread <= 20:
            return 50   # Marginal
        elif spread <= 25:
            return 30   # Poor
        else:
            return 10   # Very poor
    
    def _calculate_structure_score(self, signal):
        """Calculate score based on market structure (0-100)"""
        score = 50  # Neutral
        
        # BOS (Break of Structure) - support multiple field names
        if self._truthy(signal.get('bos')) or self._truthy(signal.get('bos_up')) or self._truthy(signal.get('bos_down')):
            score += 15
        
        # CHOCH (Change of Character)
        if self._truthy(signal.get('choch')):
            score += 15
        
        # Liquidity Event
        liquidity_event = signal.get('liquidity_event', 'NONE')
        if liquidity_event in ['BREAKOUT_CONFIRMED', 'LIQUIDITY_SWEEP']:
            score += 10
        elif liquidity_event in ['BREAKOUT_REJECTION', 'TRAP_BREAKOUT']:
            score -= 15
        
        # Lifecycle State - support both lifecycle_state and structure_state/behavior_label
        lifecycle = str(signal.get('lifecycle_state', signal.get('structure_state', signal.get('behavior_label', 'TREND_HEALTHY')))).upper()
        if lifecycle in ['TREND_HEALTHY', 'BREAKOUT_EXPANSION', 'TREND_UP', 'TREND_DOWN', 'BREAKOUT']:
            score += 10
        elif lifecycle in ['TREND_EXHAUSTING', 'EXIT_WARNING', 'CHOPPY', 'RANGE']:
            score -= 10
        elif lifecycle == 'FORCE_EXIT':
            score -= 30
        
        return max(0, min(100, score))
    
    def get_quality_tier(self, score):
        """Convert score to quality tier"""
        if score >= 85:
            return 'ELITE'
        elif score >= 70:
            return 'HIGH'
        elif score >= 55:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_recommendation(self, score, system="FLOW_EXP"):
        """Get trading recommendation based on score"""
        if system == "FLOW_EXP":
            # Stricter requirements for FLOW
            if score >= 75:
                return 'ALLOW', 'High quality FLOW setup'
            elif score >= 65:
                return 'ALLOW', 'Acceptable FLOW setup - reduce size'
            elif score >= 55:
                return 'REDUCE', 'Marginal setup - significantly reduce size'
            else:
                return 'BLOCK', 'Low quality - avoid'
        else:
            # ALPHA requirements
            if score >= 80:
                return 'ALLOW', 'High quality ALPHA setup'
            elif score >= 70:
                return 'ALLOW', 'Acceptable ALPHA setup'
            else:
                return 'BLOCK', 'Below ALPHA standards'
