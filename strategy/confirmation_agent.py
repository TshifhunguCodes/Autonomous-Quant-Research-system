import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_confirmations


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.setups, parse_dates=["time"])
    build_confirmations(df).to_csv(config.paths.confirmed_signals, index=False)
    logger.info("Confirmation stage saved at %s", config.paths.confirmed_signals)
