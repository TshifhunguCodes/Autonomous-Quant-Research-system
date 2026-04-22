import pandas as pd

from core.logging_utils import get_logger


logger = get_logger(__name__)


def run(config=None):
    signals_path = "data/features/signals.csv"
    trade_plan_path = "data/features/trade_plan.csv"

    if config is not None:
        signals_path = config.paths.signals
        trade_plan_path = config.paths.features_dir / "trade_plan.csv"

    df = pd.read_csv(signals_path, parse_dates=["time"])

    df["entry"] = df["close"]
    df["stop_loss"] = None
    df["take_profit"] = None
    df["rr"] = 2.0

    buy_mask = df["bias"] == "BUY"
    df.loc[buy_mask, "stop_loss"] = df["low"] - 0.5
    risk_buy = df.loc[buy_mask, "entry"] - df.loc[buy_mask, "stop_loss"]
    df.loc[buy_mask, "take_profit"] = df["entry"] + (risk_buy * 2)

    sell_mask = df["bias"] == "SELL"
    df.loc[sell_mask, "stop_loss"] = df["high"] + 0.5
    risk_sell = df.loc[sell_mask, "stop_loss"] - df.loc[sell_mask, "entry"]
    df.loc[sell_mask, "take_profit"] = df["entry"] - (risk_sell * 2)

    df.to_csv(trade_plan_path, index=False)
    logger.info("Legacy risk-engine trade plan saved at %s", trade_plan_path)
