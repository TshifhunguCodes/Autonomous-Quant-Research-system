import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_signals


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.regime_context, parse_dates=["time"])
    build_signals(df).to_csv(config.paths.signals, index=False)
    logger.info("Signal stage saved at %s", config.paths.signals)
