import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_market_state


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.zones, parse_dates=["time"])
    build_market_state(df).to_csv(config.paths.market_state, index=False)
    logger.info("Market-state stage saved at %s", config.paths.market_state)
