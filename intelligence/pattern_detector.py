"""
Candlestick Pattern Detector — Detects key reversal and continuation patterns
Adapted from ATS_US30_NAS into AQRS

Detects:
- Hammer, Shooting Star, Doji
- Bullish/Bearish Engulfing
- Morning Star, Evening Star
- Pin Bar, Inside Bar
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects candlestick patterns and adds pattern signals to the pipeline.
    """
    
    def __init__(self, config=None):
        self.config = config
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns and add pattern columns to DataFrame.
        
        Args:
            df: OHLCV DataFrame (must have: open, high, low, close)
        
        Returns:
            DataFrame with pattern columns added
        """
        out = df.copy()
        
        out["body"] = abs(out["close"] - out["open"])
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
        out["total_range"] = out["high"] - out["low"]
        out["is_bullish"] = (out["close"] > out["open"]).astype(int)
        out["is_bearish"] = (out["close"] < out["open"]).astype(int)
        
        # Small body threshold (percentage of total range)
        out["small_body"] = out["body"] < out["total_range"] * 0.3
        out["large_body"] = out["body"] > out["total_range"] * 0.7
        
        # === SINGLE CANDLE PATTERNS ===
        
        # Hammer: small body at top, long lower wick (2x body), little/no upper wick
        out["hammer"] = (
            out["is_bullish"]
            & out["small_body"]
            & (out["lower_wick"] > out["body"] * 2)
            & (out["upper_wick"] < out["body"] * 0.3)
        ).astype(int)
        
        # Shooting Star: small body at bottom, long upper wick (2x body), little/no lower wick
        out["shooting_star"] = (
            out["is_bearish"]
            & out["small_body"]
            & (out["upper_wick"] > out["body"] * 2)
            & (out["lower_wick"] < out["body"] * 0.3)
        ).astype(int)
        
        # Doji: very small body (less than 10% of range)
        out["doji"] = (out["body"] < out["total_range"] * 0.1).astype(int)
        
        # Pin Bar: long wick (3x body) on either side
        out["pin_bar"] = (
            ((out["upper_wick"] > out["body"] * 3) & (out["lower_wick"] < out["body"] * 0.5))
            | ((out["lower_wick"] > out["body"] * 3) & (out["upper_wick"] < out["body"] * 0.5))
        ).astype(int)
        
        # Marubozu: very long body, little to no wicks
        out["marubozu"] = (
            out["large_body"]
            & (out["upper_wick"] < out["total_range"] * 0.05)
            & (out["lower_wick"] < out["total_range"] * 0.05)
        ).astype(int)
        
        # === TWO CANDLE PATTERNS ===
        
        # Bullish Engulfing: current bullish candle fully engulfs previous bearish candle
        prev_open = out["open"].shift(1)
        prev_close = out["close"].shift(1)
        prev_high = out["high"].shift(1)
        prev_low = out["low"].shift(1)
        
        out["bullish_engulfing"] = (
            out["is_bullish"]
            & (prev_close < prev_open)  # Previous was bearish
            & (out["open"] < prev_close)  # Opens below prev close
            & (out["close"] > prev_open)  # Closes above prev open
        ).astype(int)
        
        # Bearish Engulfing: current bearish candle fully engulfs previous bullish candle
        out["bearish_engulfing"] = (
            out["is_bearish"]
            & (prev_close > prev_open)  # Previous was bullish
            & (out["open"] > prev_close)  # Opens above prev close
            & (out["close"] < prev_open)  # Closes below prev open
        ).astype(int)
        
        # Inside Bar: current range is within previous range
        out["inside_bar"] = (
            (out["high"] <= prev_high)
            & (out["low"] >= prev_low)
        ).astype(int)
        
        # === THREE CANDLE PATTERNS ===
        
        # Morning Star: bearish → doji/hammer → bullish (gap down then reversal)
        prev2_open = out["open"].shift(2)
        prev2_close = out["close"].shift(2)
        prev2_low = out["low"].shift(2)
        
        out["morning_star"] = (
            (prev2_close < prev2_open)  # 2 bars ago: bearish
            & out["small_body"].shift(1)  # 1 bar ago: small body (doji/hammer)
            & out["is_bullish"]  # Current: bullish
            & (out["close"] > (prev2_open + prev2_close) / 2)  # Closes above midpoint of first candle
        ).astype(int)
        
        # Evening Star: bullish → doji/shooting star → bearish (gap up then reversal)
        prev2_high = out["high"].shift(2)
        
        out["evening_star"] = (
            (prev2_close > prev2_open)  # 2 bars ago: bullish
            & out["small_body"].shift(1)  # 1 bar ago: small body
            & out["is_bearish"]  # Current: bearish
            & (out["close"] < (prev2_open + prev2_close) / 2)  # Closes below midpoint of first candle
        ).astype(int)
        
        # === COMPOSITE PATTERNS ===
        
        # Bullish reversal: hammer OR bullish engulfing OR morning star
        out["bullish_reversal"] = (
            (out["hammer"] == 1)
            | (out["bullish_engulfing"] == 1)
            | (out["morning_star"] == 1)
            | (out["pin_bar"] == 1) & (out["lower_wick"] > out["upper_wick"])
        ).astype(int)
        
        # Bearish reversal: shooting star OR bearish engulfing OR evening star
        out["bearish_reversal"] = (
            (out["shooting_star"] == 1)
            | (out["bearish_engulfing"] == 1)
            | (out["evening_star"] == 1)
            | (out["pin_bar"] == 1) & (out["upper_wick"] > out["lower_wick"])
        ).astype(int)
        
        # Continuation pattern: inside bar (potential breakout)
        out["continuation_pattern"] = out["inside_bar"]
        
        # Pattern score: how many bullish/bearish signals are present
        out["bullish_pattern_score"] = (
            out["hammer"]
            + out["bullish_engulfing"]
            + out["morning_star"]
            + (out["pin_bar"] & (out["lower_wick"] > out["upper_wick"])).astype(int)
        )
        
        out["bearish_pattern_score"] = (
            out["shooting_star"]
            + out["bearish_engulfing"]
            + out["evening_star"]
            + (out["pin_bar"] & (out["upper_wick"] > out["lower_wick"])).astype(int)
        )
        
        # Clean up intermediate columns (keep only pattern columns)
        pattern_cols = [
            "hammer", "shooting_star", "doji", "pin_bar", "marubozu",
            "bullish_engulfing", "bearish_engulfing", "inside_bar",
            "morning_star", "evening_star",
            "bullish_reversal", "bearish_reversal", "continuation_pattern",
            "bullish_pattern_score", "bearish_pattern_score",
        ]
        
        # Drop helper columns to keep DataFame clean
        helper_cols = ["body", "upper_wick", "lower_wick", "total_range", 
                       "small_body", "large_body", "is_bullish", "is_bearish"]
        out = out.drop(columns=[c for c in helper_cols if c in out.columns], errors="ignore")
        
        logger.info(f"Pattern detection complete. Last 5 candles:")
        logger.info(f"  Bullish patterns: {out['bullish_reversal'].iloc[-5:].sum()}")
        logger.info(f"  Bearish patterns: {out['bearish_reversal'].iloc[-5:].sum()}")
        
        return out
    
    def enrich_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern columns to pipeline DataFrame.
        Called during pipeline execution.
        """
        return self.detect(df)
