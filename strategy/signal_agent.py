import pandas as pd
import numpy as np

def run():
    df = pd.read_csv("data/features/market_state.csv")

    df["score"] = 0

    # Trend
    df.loc[df["trend"] == "bullish", "score"] += 20
    df.loc[df["trend"] == "bearish", "score"] += 20

    # Zone proximity
    df.loc[df["near_support"] == 1, "score"] += 20
    df.loc[df["near_resistance"] == 1, "score"] += 20

    # Market state
    df.loc[df["market_state"] == "TRENDING", "score"] += 15
    df.loc[df["market_state"] == "RANGING", "score"] += 8
    df.loc[df["market_state"] == "VOLATILE", "score"] += 5
    df.loc[df["market_state"] == "CHOPPY", "score"] += 0

    # Optional volume feature if exists
    if "volume_spike" in df.columns:
        df.loc[df["volume_spike"] == True, "score"] += 7

    # Candle rejection proxy
    lower_wick = df["open"].combine(df["close"], min) - df["low"]
    upper_wick = df["high"] - df["open"].combine(df["close"], max)

    df.loc[lower_wick > (df["high"] - df["low"]) * 0.4, "score"] += 10
    df.loc[upper_wick > (df["high"] - df["low"]) * 0.4, "score"] += 10

    # Session placeholder (can improve later)
    if "hour" in df.columns:
        df.loc[df["hour"].between(7, 16), "score"] += 8

    # Label result
    df["signal"] = "NO_TRADE"

    df.loc[df["score"] >= 80, "signal"] = "A_SETUP"
    df.loc[(df["score"] >= 65) & (df["score"] < 80), "signal"] = "B_SETUP"
    df.loc[(df["score"] >= 50) & (df["score"] < 65), "signal"] = "C_SETUP"

    # Direction
    df["bias"] = "NONE"
    df.loc[(df["near_support"] == 1) & (df["trend"] == "bullish"), "bias"] = "BUY"
    df.loc[(df["near_resistance"] == 1) & (df["trend"] == "bearish"), "bias"] = "SELL"

    df.to_csv("data/features/signals.csv", index=False)

    print("Scoring Signal Agent Complete")