import pandas as pd

def run():
    df = pd.read_csv("data/features/signals.csv")

    df["entry"] = df["close"]
    df["stop_loss"] = None
    df["take_profit"] = None
    df["rr"] = 2.0   # target 1:2 RR

    # BUY setups
    buy_mask = df["bias"] == "BUY"
    df.loc[buy_mask, "stop_loss"] = df["low"] - 0.5
    risk_buy = df.loc[buy_mask, "entry"] - df.loc[buy_mask, "stop_loss"]
    df.loc[buy_mask, "take_profit"] = df.loc[buy_mask, "entry"] + (risk_buy * 2)

    # SELL setups
    sell_mask = df["bias"] == "SELL"
    df.loc[sell_mask, "stop_loss"] = df["high"] + 0.5
    risk_sell = df.loc[sell_mask, "stop_loss"] - df.loc[sell_mask, "entry"]
    df.loc[sell_mask, "take_profit"] = df.loc[sell_mask, "entry"] - (risk_sell * 2)

    df.to_csv("data/features/trade_plan.csv", index=False)

    print("Risk Engine Complete")