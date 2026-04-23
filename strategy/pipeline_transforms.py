import numpy as np
import pandas as pd


def build_m5_features(df_m5: pd.DataFrame) -> pd.DataFrame:
    df = df_m5.sort_values("time").reset_index(drop=True).copy()
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["MA20"] = df["close"].rolling(20).mean()
    df["range"] = df["high"] - df["low"]
    df["momentum"] = df["close"].diff()
    df["vol_avg"] = df["tick_volume"].rolling(20).mean()
    df["volume_spike"] = df["tick_volume"] > df["vol_avg"]
    
    # Calculate ATR
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs()
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()
    
    return df


def build_h1_context(df_h1: pd.DataFrame) -> pd.DataFrame:
    df = df_h1.sort_values("time").reset_index(drop=True).copy()
    df["h1_prev_high"] = df["high"].shift(1)
    df["h1_prev_low"] = df["low"].shift(1)
    df["h1_bos_up"] = (df["high"] > df["h1_prev_high"]).astype(int)
    df["h1_bos_down"] = (df["low"] < df["h1_prev_low"]).astype(int)
    df["h1_ma20"] = df["close"].rolling(20).mean()
    df["h1_momentum"] = df["close"].diff()
    df["h1_trend"] = "neutral"
    df["h1_trend"] = np.where(
        df["h1_bos_up"] == 1,
        "bullish",
        np.where(df["h1_bos_down"] == 1, "bearish", "neutral"),
    )

    bullish_bias = (
        (df["close"] > df["h1_ma20"]) & (df["h1_momentum"] >= 0)
    ) | (df["h1_trend"] == "bullish")
    bearish_bias = (
        (df["close"] < df["h1_ma20"]) & (df["h1_momentum"] <= 0)
    ) | (df["h1_trend"] == "bearish")

    df["h1_bias"] = "neutral"
    df.loc[bullish_bias, "h1_bias"] = "bullish"
    df.loc[bearish_bias, "h1_bias"] = "bearish"

    return df[
        [
            "time",
            "h1_prev_high",
            "h1_prev_low",
            "h1_bos_up",
            "h1_bos_down",
            "h1_ma20",
            "h1_momentum",
            "h1_trend",
            "h1_bias",
        ]
    ]


def merge_h1_context_into_m5(
    df_m5_features: pd.DataFrame, df_h1: pd.DataFrame
) -> pd.DataFrame:
    h1_context = build_h1_context(df_h1)
    return pd.merge_asof(
        df_m5_features.sort_values("time"),
        h1_context.sort_values("time"),
        on="time",
        direction="backward",
    )


def build_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prev_high"] = out["high"].shift(1)
    out["prev_low"] = out["low"].shift(1)
    out["bos_up"] = (out["high"] > out["prev_high"]).astype(int)
    out["bos_down"] = (out["low"] < out["prev_low"]).astype(int)
    out["trend"] = "neutral"
    out["trend"] = np.where(
        out["bos_up"] == 1,
        "bullish",
        np.where(out["bos_down"] == 1, "bearish", "neutral"),
    )
    return out


def build_zones(df: pd.DataFrame, config) -> pd.DataFrame:
    out = df.copy()
    out["zone_high_20"] = out["high"].rolling(20).max()
    out["zone_low_20"] = out["low"].rolling(20).min()
    out["zone_high_50"] = out["high"].rolling(50).max()
    out["zone_low_50"] = out["low"].rolling(50).min()

    out["near_support"] = (
        abs(out["close"] - out["zone_low_20"]) <= config.zones.near_threshold
    ).astype(int)
    out["near_resistance"] = (
        abs(out["close"] - out["zone_high_20"]) <= config.zones.near_threshold
    ).astype(int)
    out["major_support"] = (
        abs(out["close"] - out["zone_low_50"]) <= config.zones.major_threshold
    ).astype(int)
    out["major_resistance"] = (
        abs(out["close"] - out["zone_high_50"]) <= config.zones.major_threshold
    ).astype(int)
    return out


def build_market_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["candle_range"] = out["high"] - out["low"]
    out["avg_range_20"] = out["candle_range"].rolling(20).mean()
    out["direction"] = (out["close"] > out["open"]).astype(int)
    out["flip"] = (out["direction"] != out["direction"].shift(1)).astype(int)
    out["flip_count_10"] = out["flip"].rolling(10).sum()
    out["market_state"] = "RANGING"
    out.loc[
        (out["flip_count_10"] <= 3)
        & (out["candle_range"] > out["avg_range_20"]),
        "market_state",
    ] = "TRENDING"
    out.loc[out["flip_count_10"] >= 6, "market_state"] = "CHOPPY"
    out.loc[
        out["candle_range"] > out["avg_range_20"] * 2,
        "market_state",
    ] = "VOLATILE"
    return out


def build_regime_layer(df: pd.DataFrame, config) -> pd.DataFrame:
    out = df.copy()
    if "h1_alignment" not in out.columns:
        out["h1_alignment"] = 0
        if "h1_bias" in out.columns and "trend" in out.columns:
            bullish_alignment = (
                (out["trend"] == "bullish") & (out["h1_bias"] == "bullish")
            )
            bearish_alignment = (
                (out["trend"] == "bearish") & (out["h1_bias"] == "bearish")
            )
            out.loc[bullish_alignment | bearish_alignment, "h1_alignment"] = 1

    out["market_regime"] = "NEUTRAL"
    out.loc[out["market_state"] == "CHOPPY", "market_regime"] = "CHOPPY"
    out.loc[out["market_state"] == "VOLATILE", "market_regime"] = "VOLATILE"
    out.loc[
        (out["market_state"] == "TRENDING") & (out["h1_alignment"] == 1),
        "market_regime",
    ] = "ALIGNED_TREND"
    out.loc[
        (out["market_state"] == "RANGING") & (out["h1_alignment"] == 1),
        "market_regime",
    ] = "ALIGNED_RANGE"
    out.loc[
        (out["market_state"] == "TRENDING") & (out["h1_alignment"] == 0),
        "market_regime",
    ] = "TREND_MISMATCH"

    risk_map = {
        "ALIGNED_RANGE": config.regime.aligned_range_risk_multiplier,
        "ALIGNED_TREND": config.regime.aligned_trend_risk_multiplier,
        "NEUTRAL": config.regime.neutral_risk_multiplier,
        "TREND_MISMATCH": config.regime.trend_mismatch_risk_multiplier,
        "VOLATILE": config.regime.volatile_risk_multiplier,
        "CHOPPY": config.regime.choppy_risk_multiplier,
    }
    out["regime_risk_multiplier"] = (
        out["market_regime"].map(risk_map).fillna(config.regime.neutral_risk_multiplier)
    )
    out["regime_trade_band"] = "FAVORABLE"
    out.loc[
        out["market_regime"].isin(["VOLATILE", "TREND_MISMATCH", "NEUTRAL"]),
        "regime_trade_band",
    ] = "CAUTION"
    out.loc[out["market_regime"] == "CHOPPY", "regime_trade_band"] = "DEFENSIVE"
    return out


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = 0
    out.loc[out["trend"] == "bullish", "score"] += 20
    out.loc[out["trend"] == "bearish", "score"] += 20
    out.loc[out["near_support"] == 1, "score"] += 20
    out.loc[out["near_resistance"] == 1, "score"] += 20
    out.loc[out["market_state"] == "TRENDING", "score"] += 15
    out.loc[out["market_state"] == "RANGING", "score"] += 8
    out.loc[out["market_state"] == "VOLATILE", "score"] += 5

    if "volume_spike" in out.columns:
        out.loc[out["volume_spike"] == True, "score"] += 7

    lower_wick = out["open"].combine(out["close"], min) - out["low"]
    upper_wick = out["high"] - out["open"].combine(out["close"], max)
    out.loc[lower_wick > (out["high"] - out["low"]) * 0.4, "score"] += 10
    out.loc[upper_wick > (out["high"] - out["low"]) * 0.4, "score"] += 10

    if "hour" in out.columns:
        out.loc[out["hour"].between(7, 16), "score"] += 8

    out["h1_alignment"] = 0
    if "h1_bias" in out.columns:
        bullish_alignment = (
            (out["trend"] == "bullish") & (out["h1_bias"] == "bullish")
        )
        bearish_alignment = (
            (out["trend"] == "bearish") & (out["h1_bias"] == "bearish")
        )
        out.loc[bullish_alignment | bearish_alignment, "h1_alignment"] = 1
        out.loc[out["h1_alignment"] == 1, "score"] += 12

    out["signal"] = "NO_TRADE"
    out.loc[out["score"] >= 80, "signal"] = "A_SETUP"
    out.loc[(out["score"] >= 65) & (out["score"] < 80), "signal"] = "B_SETUP"
    out.loc[(out["score"] >= 50) & (out["score"] < 65), "signal"] = "C_SETUP"

    buy_bias = (out["near_support"] == 1) & (out["trend"] == "bullish")
    sell_bias = (out["near_resistance"] == 1) & (out["trend"] == "bearish")

    if "h1_bias" in out.columns:
        buy_bias &= out["h1_bias"] == "bullish"
        sell_bias &= out["h1_bias"] == "bearish"

    out["bias"] = "NONE"
    out.loc[buy_bias, "bias"] = "BUY"
    out.loc[sell_bias, "bias"] = "SELL"
    return out


def build_setups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["setup"] = "NONE"
    out["setup_score"] = 0
    out["buy_count"] = (out["bias"] == "BUY").rolling(3).sum()
    out["sell_count"] = (out["bias"] == "SELL").rolling(3).sum()

    buy_mask = (out["bias"] == "BUY") & (out["signal"] != "NO_TRADE")
    out.loc[buy_mask & (out["buy_count"] >= 2), "setup_score"] += 20
    if "near_support" in out.columns:
        out.loc[buy_mask & (out["near_support"] == 1), "setup_score"] += 25
    if "trend" in out.columns:
        out.loc[buy_mask & (out["trend"] == "bullish"), "setup_score"] += 25
    if "h1_alignment" in out.columns:
        out.loc[buy_mask & (out["h1_alignment"] == 1), "setup_score"] += 15
    out.loc[buy_mask & ((out["high"] - out["low"]) > 1.5), "setup_score"] += 15
    out.loc[buy_mask & (out["setup_score"] >= 50), "setup"] = "BUY_SETUP"

    sell_mask = (out["bias"] == "SELL") & (out["signal"] != "NO_TRADE")
    out.loc[sell_mask & (out["sell_count"] >= 2), "setup_score"] += 20
    if "near_resistance" in out.columns:
        out.loc[sell_mask & (out["near_resistance"] == 1), "setup_score"] += 25
    if "trend" in out.columns:
        out.loc[sell_mask & (out["trend"] == "bearish"), "setup_score"] += 25
    if "h1_alignment" in out.columns:
        out.loc[sell_mask & (out["h1_alignment"] == 1), "setup_score"] += 15
    out.loc[sell_mask & ((out["high"] - out["low"]) > 1.5), "setup_score"] += 15
    out.loc[sell_mask & (out["setup_score"] >= 50), "setup"] = "SELL_SETUP"
    return out


def classify_trade(score):
    if score >= 70:
        return "ELITE"
    if score >= 60:
        return "HIGH"
    if score >= 55:
        return "MEDIUM"
    return "NO_TRADE"


def build_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["body"] = abs(out["close"] - out["open"])
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["is_bullish"] = out["close"] > out["open"]
    out["is_bearish"] = out["close"] < out["open"]
    out["confirm_score"] = 0

    # Climax Detection: If candle range is 2x the average, it's likely exhaustion
    out["is_climax"] = (out["high"] - out["low"]) > (out["avg_range_20"] * 2.0)

    # Reverting Anti-Indecision to 30% for better entry participation
    out["is_indecision"] = out["body"] < (out["high"] - out["low"]) * 0.3

    out["confirmed_signal"] = "no_trade"
    buy_mask = (out["setup"] == "BUY_SETUP") & (out["near_support"] == 1)
    sell_mask = (out["setup"] == "SELL_SETUP") & (out["near_resistance"] == 1)

    if "h1_alignment" in out.columns:
        buy_mask &= out["h1_alignment"] == 1
        sell_mask &= out["h1_alignment"] == 1

    # Only reward wicks if the body isn't anemic
    out.loc[buy_mask & (out["lower_wick"] > out["body"]) & (~out["is_indecision"]), "confirm_score"] += 20
    
    # Softening penalties to prevent late entries / confirmation trap
    out.loc[buy_mask & out["is_indecision"], "confirm_score"] -= 15
    out.loc[buy_mask & out["is_climax"], "confirm_score"] -= 20
    
    out.loc[buy_mask & out["is_bullish"], "confirm_score"] += 25
    if "trend" in out.columns:
        out.loc[buy_mask & (out["trend"] == "bullish"), "confirm_score"] += 20
    if "market_state" in out.columns:
        out.loc[
            buy_mask
            & (out["market_state"] == "RANGING")
            & (out["major_support"] == 1),
            "confirm_score",
        ] += 15
    out.loc[buy_mask & (out["major_support"] == 1), "confirm_score"] += 20
    if "h1_bias" in out.columns:
        out.loc[buy_mask & (out["h1_bias"] == "bullish"), "confirm_score"] += 10
    out.loc[
        (out["confirm_score"] >= 55) & (out["setup"] == "BUY_SETUP"),
        "confirmed_signal",
    ] = "buy"

    out.loc[sell_mask & (out["upper_wick"] > out["body"]) & (~out["is_indecision"]), "confirm_score"] += 20
    out.loc[sell_mask & out["is_indecision"], "confirm_score"] -= 15
    out.loc[sell_mask & out["is_climax"], "confirm_score"] -= 20
    
    out.loc[sell_mask & out["is_bearish"], "confirm_score"] += 25
    if "trend" in out.columns:
        out.loc[sell_mask & (out["trend"] == "bearish"), "confirm_score"] += 20
    if "market_state" in out.columns:
        out.loc[
            sell_mask
            & (out["market_state"] == "RANGING")
            & (out["major_resistance"] == 1),
            "confirm_score",
        ] += 15
    out.loc[sell_mask & (out["major_resistance"] == 1), "confirm_score"] += 20
    if "h1_bias" in out.columns:
        out.loc[sell_mask & (out["h1_bias"] == "bearish"), "confirm_score"] += 10
    out.loc[
        (out["confirm_score"] >= 55) & (out["setup"] == "SELL_SETUP"),
        "confirmed_signal",
    ] = "sell"

    out["quality"] = out["confirm_score"].apply(classify_trade)
    return out


def build_trade_setups(df: pd.DataFrame, config) -> pd.DataFrame:
    out = df.copy()
    base_offset = config.risk.base_stop_buffer
    rr_ratio = config.risk.rr_ratio
    state_multiplier = {
        "TRENDING": 0.9,
        "RANGING": 1.0,
        "VOLATILE": 1.4,
        "CHOPPY": 1.2,
    }

    out["rr_ratio"] = rr_ratio
    out["stop_buffer"] = (
        out["market_state"].map(state_multiplier).fillna(1.0) * base_offset
    )
    out["entry_price"] = np.nan
    out["stop_loss"] = np.nan
    out["take_profit"] = np.nan
    out["risk_distance"] = np.nan
    out["quality_risk_multiplier"] = 1.0
    out.loc[out["quality"] == "MEDIUM", "quality_risk_multiplier"] = (
        config.regime.medium_quality_risk_multiplier
    )
    out.loc[out["quality"] == "HIGH", "quality_risk_multiplier"] = (
        config.regime.high_quality_risk_multiplier
    )
    out.loc[out["quality"] == "ELITE", "quality_risk_multiplier"] = (
        config.regime.elite_quality_risk_multiplier
    )
    out["risk_multiplier"] = (
        out.get("regime_risk_multiplier", pd.Series(1.0, index=out.index))
        * out["quality_risk_multiplier"]
    )
    out["trade_allowed"] = out["confirmed_signal"].isin(["buy", "sell"])
    out["risk_dampening_reason"] = ""

    choppy_block = (
        config.regime.block_choppy_non_elite
        & (out["market_regime"] == "CHOPPY")
        & (out["quality"] != "ELITE")
        & out["trade_allowed"]
    )
    out.loc[choppy_block, "trade_allowed"] = False
    out.loc[choppy_block, "risk_dampening_reason"] = "blocked_choppy_non_elite"

    volatile_medium_block = (
        config.regime.block_volatile_medium
        & (out["market_regime"] == "VOLATILE")
        & (out["quality"] == "MEDIUM")
        & out["trade_allowed"]
    )
    out.loc[volatile_medium_block, "trade_allowed"] = False
    out.loc[
        volatile_medium_block, "risk_dampening_reason"
    ] = "blocked_volatile_medium"

    # Session-aware filter
    if hasattr(config, 'session_filters') and config.session_filters.disable_late_session:
        late_session_mask = (out["hour"] >= config.session_filters.late_session_start_hour) & out["trade_allowed"]
        out.loc[late_session_mask, "trade_allowed"] = False
        out.loc[late_session_mask, "risk_dampening_reason"] = "late_session_disabled"

    # Disabled sessions filter
    def get_session(hour):
        if 0 <= hour < 8:
            return 'Asian'
        elif 8 <= hour < 16:
            return 'European'
        else:
            return 'US'

    out['session'] = out['hour'].apply(get_session)
    if hasattr(config, 'session_filters') and config.session_filters.disabled_sessions:
        session_mask = out['session'].isin(config.session_filters.disabled_sessions) & out["trade_allowed"]
        out.loc[session_mask, "trade_allowed"] = False
        out.loc[session_mask, "risk_dampening_reason"] = "disabled_session"

    # Disabled market states filter
    if hasattr(config, 'session_filters') and config.session_filters.disabled_market_states:
        state_mask = out['market_state'].isin(config.session_filters.disabled_market_states) & out["trade_allowed"]
        out.loc[state_mask, "trade_allowed"] = False
        out.loc[state_mask, "risk_dampening_reason"] = "disabled_market_state"

    buy_mask = (out["confirmed_signal"] == "buy") & out["trade_allowed"]
    sell_mask = (out["confirmed_signal"] == "sell") & out["trade_allowed"]
    out.loc[buy_mask, "entry_price"] = out["close"]
    out.loc[buy_mask, "stop_loss"] = out["low"] - out["stop_buffer"]
    buy_risk = out["close"] - (out["low"] - out["stop_buffer"])
    out.loc[buy_mask, "risk_distance"] = buy_risk
    out.loc[buy_mask, "take_profit"] = out["close"] + (buy_risk * rr_ratio)

    out.loc[sell_mask, "entry_price"] = out["close"]
    out.loc[sell_mask, "stop_loss"] = out["high"] + out["stop_buffer"]
    sell_risk = (out["high"] + out["stop_buffer"]) - out["close"]
    out.loc[sell_mask, "risk_distance"] = sell_risk
    out.loc[sell_mask, "take_profit"] = out["close"] - (sell_risk * rr_ratio)
    return out


def run_strategy_pipeline(
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame,
    config,
    return_stages: bool = False,
):
    features = build_m5_features(df_m5)
    with_h1 = merge_h1_context_into_m5(features, df_h1)
    structure = build_structure(with_h1)
    zones = build_zones(structure, config)
    market_state = build_market_state(zones)
    regime = build_regime_layer(market_state, config)
    signals = build_signals(regime)
    setups = build_setups(signals)
    confirmations = build_confirmations(setups)
    trade_setups = build_trade_setups(confirmations, config)

    if return_stages:
        return {
            "features": features,
            "with_h1": with_h1,
            "structure": structure,
            "zones": zones,
            "market_state": market_state,
            "regime": regime,
            "signals": signals,
            "setups": setups,
            "confirmations": confirmations,
            "trade_setups": trade_setups,
        }
    return trade_setups
