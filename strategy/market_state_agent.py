import pandas as pd

def run():
    df = pd.read_csv("data/features/zones.csv")

    # Candle range
    df["candle_range"] = df["high"] - df["low"]

    # Average recent range
    df["avg_range_20"] = df["candle_range"].rolling(20).mean()

    # Direction change
    df["direction"] = (df["close"] > df["open"]).astype(int)
    df["flip"] = (df["direction"] != df["direction"].shift(1)).astype(int)

    # Flip count last 10 candles
    df["flip_count_10"] = df["flip"].rolling(10).sum()

    # Default state
    df["market_state"] = "RANGING"

    # Trending = low flips + decent range
    df.loc[
        (df["flip_count_10"] <= 3) &
        (df["candle_range"] > df["avg_range_20"]),
        "market_state"
    ] = "TRENDING"

    # Choppy = many flips
    df.loc[
        df["flip_count_10"] >= 6,
        "market_state"
    ] = "CHOPPY"

    # Volatile = huge candle
    df.loc[
        df["candle_range"] > df["avg_range_20"] * 2,
        "market_state"
    ] = "VOLATILE"

    df.to_csv("data/features/market_state.csv", index=False)

    print("Market State Agent Complete")