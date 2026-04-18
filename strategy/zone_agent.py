import pandas as pd

def run():
    df = pd.read_csv("data/features/structure.csv")

    # rolling zones
    df["zone_high_20"] = df["high"].rolling(20).max()
    df["zone_low_20"] = df["low"].rolling(20).min()

    # near support if close near rolling low
    df["near_support"] = (
        abs(df["close"] - df["zone_low_20"]) < 1.0
    ).astype(int)

    # near resistance if close near rolling high
    df["near_resistance"] = (
        abs(df["close"] - df["zone_high_20"]) < 1.0
    ).astype(int)

    df.to_csv("data/features/zones.csv", index=False)

    print("Zone Agent Complete")