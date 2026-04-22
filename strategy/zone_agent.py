import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_zones


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.structure, parse_dates=["time"])
    build_zones(df, config).to_csv(config.paths.zones, index=False)
    logger.info("Zone stage saved at %s", config.paths.zones)
