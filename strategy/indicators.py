"""
Additional Technical Indicators — MACD, Bollinger Bands, ADX, Stochastic
Adapted from ATS_US30_NAS into AQRS

Adds to the existing indicator set (ATR, RSI, EMA) with:
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (BB)
- ADX (Average Directional Index)
- Stochastic Oscillator
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Computes additional technical indicators and adds them to the pipeline.
    Designed to be called during pipeline execution.
    """
    
    def __init__(self, config=None):
        self.config = config
    
    def compute_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Compute MACD indicator.
        
        Returns:
            DataFrame with: macd, macd_signal, macd_histogram, macd_crossover
        """
        result = pd.DataFrame(index=close.index)
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        result["macd"] = ema_fast - ema_slow
        result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]
        
        # Crossover signals
        result["macd_crossover"] = 0
        result.loc[(result["macd"] > result["macd_signal"]) & (result["macd"].shift(1) <= result["macd_signal"].shift(1)), "macd_crossover"] = 1  # Bullish
        result.loc[(result["macd"] < result["macd_signal"]) & (result["macd"].shift(1) >= result["macd_signal"].shift(1)), "macd_crossover"] = -1  # Bearish
        
        # Zero-line cross (momentum shift)
        result["macd_zero_cross"] = 0
        result.loc[(result["macd"] > 0) & (result["macd"].shift(1) <= 0), "macd_zero_cross"] = 1
        result.loc[(result["macd"] < 0) & (result["macd"].shift(1) >= 0), "macd_zero_cross"] = -1
        
        # MACD slope (acceleration)
        result["macd_slope"] = result["macd"].diff()
        
        return result
    
    def compute_bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Compute Bollinger Bands.
        
        Returns:
            DataFrame with: bb_middle, bb_upper, bb_lower, bb_width, bb_position, bb_squeeze
        """
        result = pd.DataFrame(index=close.index)
        
        result["bb_middle"] = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        result["bb_upper"] = result["bb_middle"] + (std * std_dev)
        result["bb_lower"] = result["bb_middle"] - (std * std_dev)
        result["bb_width"] = result["bb_upper"] - result["bb_lower"]
        
        # Position within bands (0 = at lower, 1 = at upper, 0.5 = middle)
        band_range = result["bb_upper"] - result["bb_lower"]
        result["bb_position"] = (close - result["bb_lower"]) / band_range.replace(0, np.nan)
        result["bb_position"] = result["bb_position"].clip(0, 1)
        
        # Squeeze (bands narrowing = potential breakout)
        result["bb_squeeze"] = (result["bb_width"] < result["bb_width"].rolling(20).mean() * 0.8).astype(int)
        
        # Touch signals
        result["bb_touch_upper"] = (close >= result["bb_upper"] * 0.995).astype(int)
        result["bb_touch_lower"] = (close <= result["bb_lower"] * 1.005).astype(int)
        
        return result
    
    def compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        Compute Average Directional Index (ADX) and Directional Movement (DI+ / DI-).
        
        Returns:
            DataFrame with: adx, adx_plus_di, adx_minus_di, adx_strength
        """
        result = pd.DataFrame(index=close.index)
        
        # True Range
        prev_close = close.shift(1)
        tr = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs())
        )
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth using Wilder's method (EMA with alpha = 1/period)
        alpha = 1.0 / period
        tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        
        # Directional Indicators
        result["adx_plus_di"] = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
        result["adx_minus_di"] = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
        
        # Directional Index
        dx = 100 * abs(result["adx_plus_di"] - result["adx_minus_di"]) / (result["adx_plus_di"] + result["adx_minus_di"]).replace(0, np.nan)
        result["adx"] = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # ADX strength classification
        result["adx_strength"] = "WEAK"
        result.loc[result["adx"] >= 25, "adx_strength"] = "MODERATE"
        result.loc[result["adx"] >= 40, "adx_strength"] = "STRONG"
        result.loc[result["adx"] >= 60, "adx_strength"] = "VERY_STRONG"
        
        # DI crossover signals
        result["adx_bullish_cross"] = ((result["adx_plus_di"] > result["adx_minus_di"]) & 
                                        (result["adx_plus_di"].shift(1) <= result["adx_minus_di"].shift(1))).astype(int)
        result["adx_bearish_cross"] = ((result["adx_plus_di"] < result["adx_minus_di"]) & 
                                        (result["adx_plus_di"].shift(1) >= result["adx_minus_di"].shift(1))).astype(int)
        
        return result
    
    def compute_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Compute Stochastic Oscillator.
        
        Returns:
            DataFrame with: stoch_k, stoch_d, stoch_overbought, stoch_oversold, stoch_crossover
        """
        result = pd.DataFrame(index=close.index)
        
        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        low_min = low.rolling(k_period).min()
        high_max = high.rolling(k_period).max()
        
        result["stoch_k"] = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
        result["stoch_d"] = result["stoch_k"].rolling(d_period).mean()
        
        # Smooth %K (optional, using 3-period SMA)
        result["stoch_k_slow"] = result["stoch_k"].rolling(3).mean()
        
        # Levels
        result["stoch_overbought"] = (result["stoch_k"] > 80).astype(int)
        result["stoch_oversold"] = (result["stoch_k"] < 20).astype(int)
        
        # Crossover signals
        result["stoch_bullish_cross"] = ((result["stoch_k"] > result["stoch_d"]) & 
                                          (result["stoch_k"].shift(1) <= result["stoch_d"].shift(1)) &
                                          (result["stoch_k"] < 30)).astype(int)
        result["stoch_bearish_cross"] = ((result["stoch_k"] < result["stoch_d"]) & 
                                          (result["stoch_k"].shift(1) >= result["stoch_d"].shift(1)) &
                                          (result["stoch_k"] > 70)).astype(int)
        
        return result
    
    def enrich_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicator columns to the pipeline DataFrame.
        Called during pipeline execution.
        """
        out = df.copy()
        
        # MACD
        macd = self.compute_macd(out["close"])
        for col in macd.columns:
            if col not in out.columns:
                out[col] = macd[col]
        
        # Bollinger Bands
        bb = self.compute_bollinger_bands(out["close"])
        for col in bb.columns:
            if col not in out.columns:
                out[col] = bb[col]
        
        # ADX
        adx = self.compute_adx(out["high"], out["low"], out["close"])
        for col in adx.columns:
            if col not in out.columns:
                out[col] = adx[col]
        
        # Stochastic
        stoch = self.compute_stochastic(out["high"], out["low"], out["close"])
        for col in stoch.columns:
            if col not in out.columns:
                out[col] = stoch[col]
        
        logger.info(f"Added indicators: MACD (3), BB (7), ADX (5), Stoch (6)")
        
        return out