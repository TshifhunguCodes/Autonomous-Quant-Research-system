import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_trade_setups


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.confirmed_signals, parse_dates=["time"])
    build_trade_setups(df, config).to_csv(config.paths.trade_setups, index=False)
    logger.info("Trade setups saved at %s", config.paths.trade_setups)
