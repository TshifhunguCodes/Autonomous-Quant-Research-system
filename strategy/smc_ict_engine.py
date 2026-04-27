import numpy as np
import pandas as pd
from datetime import time

class SMCEngine:
    """Core SMC/ICT logic for AQRS V3: OB, FVG, Liquidity, and Kill Zones."""

    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """Detect Fair Value Gaps (3-candle imbalance)."""
        out = df.copy()
        # Require displacement: Middle candle must be significant relative to ATR
        body_size = (out["close"] - out["open"]).abs()
        vol_avg = out["tick_volume"].rolling(20).mean()
        vol_spike = out["tick_volume"] > vol_avg * 1.3

        # Bullish FVG: Low of candle 3 > High of candle 1
        out["fvg_bullish"] = (out["low"] > out["high"].shift(2)) & ((body_size.shift(1) > out["atr"].shift(1) * 1.5) | vol_spike.shift(1))
        # Bearish FVG: High of candle 3 < Low of candle 1
        out["fvg_bearish"] = (out["high"] < out["low"].shift(2)) & ((body_size.shift(1) > out["atr"].shift(1) * 1.5) | vol_spike.shift(1))
        return out

    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
        """Detect stop hunts above/below recent swing points."""
        out = df.copy()
        lookback = 20
        recent_high = out["high"].rolling(window=lookback).max().shift(1)
        recent_low = out["low"].rolling(window=lookback).min().shift(1)

        # Sweep: Price pierces high/low then closes back inside
        out["sweep_high"] = (out["high"] > recent_high) & (out["close"] <= recent_high)
        out["sweep_low"] = (out["low"] < recent_low) & (out["close"] >= recent_low)
        return out

    @staticmethod
    def identify_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """Identify Order Blocks: the last opposing candle before a structural break."""
        out = df.copy()
        out["ob_bullish"] = 0.0
        out["ob_bearish"] = 0.0
        
        # Displacement check: Instititutional size footprints (Price Move + Volume Spike)
        move_size = (out["close"] - out["close"].shift(3)).abs()
        vol_avg = out["tick_volume"].rolling(20).mean()
        vol_spike = out["tick_volume"] > vol_avg * 1.5
        
        # Significant move OR Moderate move with Heavy Volume
        displacement = (move_size > out["atr"] * 2.0) | ((move_size > out["atr"] * 1.2) & vol_spike)

        # Bullish OB: Last bearish candle before a move that creates an 'HH' structure
        bullish_ob_mask = (out["structure_label"] == "HH") & (out["close"].shift(1) < out["open"].shift(1)) & displacement
        out.loc[bullish_ob_mask, "ob_bullish"] = out["low"].shift(1)

        # Bearish OB: Last bullish candle before a move that creates an 'LL' structure
        bearish_ob_mask = (out["structure_label"] == "LL") & (out["close"].shift(1) > out["open"].shift(1)) & displacement
        out.loc[bearish_ob_mask, "ob_bearish"] = out["high"].shift(1)

        out["last_ob_bullish"] = out["ob_bullish"].replace(0, np.nan).ffill()
        out["last_ob_bearish"] = out["ob_bearish"].replace(0, np.nan).ffill()
        return out

    @staticmethod
    def calculate_premium_discount(df: pd.DataFrame) -> pd.DataFrame:
        """Institutional Midpoint: Equilibrium of the dealing range."""
        out = df.copy()
        range_high = out["high"].rolling(50).max()
        range_low = out["low"].rolling(50).min()
        midpoint = (range_high + range_low) / 2

        out["is_premium"] = out["close"] > midpoint # Expensive: Don't Buy
        out["is_discount"] = out["close"] < midpoint # Cheap: Don't Sell
        return out
    
    @staticmethod
    def detect_ote(df: pd.DataFrame) -> pd.DataFrame:
        """Detect Optimal Trade Entry (OTE) Fibonacci levels (0.62-0.79)."""
        out = df.copy()
        range_high = out["high"].rolling(50).max()
        range_low = out["low"].rolling(50).min()
        diff = range_high - range_low

        # Bullish OTE: Retracement into 62%-79% of the bullish price leg
        out["ote_bullish"] = (out["close"] <= (range_high - 0.62 * diff)) & (out["close"] >= (range_high - 0.79 * diff))
        # Bearish OTE: Retracement into 62%-79% of the bearish price leg
        out["ote_bearish"] = (out["close"] >= (range_low + 0.62 * diff)) & (out["close"] <= (range_low + 0.79 * diff))
        return out
    
    @staticmethod
    def detect_mss(df: pd.DataFrame) -> pd.DataFrame:
        """Market Structure Shift (MSS): CHoCH after a Liquidity Sweep."""
        out = df.copy()
        # MSS Bullish: Sweep Low followed by HH (Structure Break)
        out["mss_bullish"] = (out["sweep_low"].shift(5).rolling(10).max() > 0) & (out["structure_label"] == "HH")
        # MSS Bearish: Sweep High followed by LL (Structure Break)
        out["mss_bearish"] = (out["sweep_high"].shift(5).rolling(10).max() > 0) & (out["structure_label"] == "LL")
        return out

    @staticmethod
    def is_ict_kill_zone(hour: int) -> bool:
        """ICT Kill Zones: London Open and NY Open (UTC Server Time Adjusted)."""
        # For XAUUSD, these are the high-volume 'Time' windows
        london_open = 7 <= hour <= 10
        ny_open = 12 <= hour <= 15
        return london_open or ny_open

    @classmethod
    def enrich_intelligence(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich the 83-column pipeline with SMC intelligence."""
        df = cls.detect_liquidity_sweeps(df) # Run sweeps before MSS
        df = cls.detect_fvg(df)
        df = cls.identify_order_blocks(df)
        df = cls.calculate_premium_discount(df)
        df = cls.detect_ote(df)
        df = cls.detect_mss(df)
        
        # ICT Score: Confluence of SMC factors
        df["smc_score"] = 0
        df.loc[df["fvg_bullish"] | df["fvg_bearish"], "smc_score"] += 20
        df.loc[df["mss_bullish"] | df["mss_bearish"], "smc_score"] += 30
        df.loc[df["sweep_low"] | df["sweep_high"], "smc_score"] += 25
        df.loc[df["ote_bullish"] | df["ote_bearish"], "smc_score"] += 15
        df.loc[df["near_prev_poc"] == 1, "smc_score"] += 10 # Add POC to SMC score
        df.loc[(df["near_prev_vah"] == 1) | (df["near_prev_val"] == 1), "smc_score"] += 5 # VAH/VAL to SMC score
        
        # Midnight Open Anchor
        df["date"] = df["time"].dt.date
        midnight_prices = df[df["time"].dt.hour == 0].groupby("date")["open"].first()
        df["midnight_open"] = df["date"].map(midnight_prices).ffill()
        
        return df