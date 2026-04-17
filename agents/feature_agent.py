import pandas as pd

def run():
    df = pd.read_csv("data/clean/xauusd_m5_clean.csv")

    df["MA20"] = df["close"].rolling(20).mean()
    df["range"] = df["high"] - df["low"]
    df["momentum"] = df["close"].diff()

    df["vol_avg"] = df["tick_volume"].rolling(20).mean()
    df["volume_spike"] = df["tick_volume"] > df["vol_avg"]

    df.to_csv("data/features/xauusd_m5_features.csv", index=False)

    print("Features saved")