import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_structure


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.m5_features, parse_dates=["time"])
    build_structure(df).to_csv(config.paths.structure, index=False)
    logger.info("Structure stage saved at %s", config.paths.structure)
