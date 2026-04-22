import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_m5_features


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.clean_m5, parse_dates=["time"])
    build_m5_features(df).to_csv(config.paths.m5_features, index=False)
    logger.info("M5 feature set saved at %s", config.paths.m5_features)
