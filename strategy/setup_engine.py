import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_setups


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.signals, parse_dates=["time"])
    build_setups(df).to_csv(config.paths.setups, index=False)
    logger.info("Setup stage saved at %s", config.paths.setups)
