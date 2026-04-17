import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def get_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, bars=5000):
    if not mt5.initialize():
        print("MT5 failed")
        return

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df

def run():
    df_m5 = get_data("XAUUSD", mt5.TIMEFRAME_M5, 5000)
    df_h1 = get_data("XAUUSD", mt5.TIMEFRAME_H1, 2000)

    df_m5.to_csv("data/raw/xauusd_m5.csv", index=False)
    df_h1.to_csv("data/raw/xauusd_h1.csv", index=False)

    print("Raw data saved")