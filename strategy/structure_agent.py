import pandas as pd

def run():
    df = pd.read_csv("data/features/xauusd_m5_features.csv")

    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)

    df["bos_up"] = (df["high"] > df["prev_high"]).astype(int)
    df["bos_down"] = (df["low"] < df["prev_low"]).astype(int)

    df["trend"] = "neutral"

    df.loc[df["bos_up"] == 1, "trend"] = "bullish"
    df.loc[df["bos_down"] == 1, "trend"] = "bearish"

    df.to_csv("data/features/structure.csv", index=False)

    print("Structure Agent Complete")