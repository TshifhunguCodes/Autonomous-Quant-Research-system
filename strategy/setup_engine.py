import pandas as pd

def run():
    df = pd.read_csv("data/features/signals.csv")

    # default setup state
    df["setup"] = "NONE"

    # detect repeated BUY signals (setup formation)
    df["buy_count"] = (df["bias"] == "BUY").rolling(3).sum()
    df["sell_count"] = (df["bias"] == "SELL").rolling(3).sum()

    # valid BUY setup = repeated strength
    df.loc[df["buy_count"] >= 2, "setup"] = "BUY_SETUP"

    # valid SELL setup
    df.loc[df["sell_count"] >= 2, "setup"] = "SELL_SETUP"

    # remove noise
    df.loc[df["signal"] == "NO_TRADE", "setup"] = "NONE"

    df.to_csv("data/features/setups.csv", index=False)

    print("Setup Engine Complete")