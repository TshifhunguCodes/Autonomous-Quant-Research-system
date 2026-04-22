import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import merge_h1_context_into_m5


logger = get_logger(__name__)
def run(config):
    m5 = pd.read_csv(config.paths.m5_features, parse_dates=["time"])
    h1 = pd.read_csv(config.paths.clean_h1, parse_dates=["time"])

    merge_h1_context_into_m5(m5, h1).to_csv(config.paths.m5_features, index=False)
    logger.info(
        "Multi-timeframe context merged into %s",
        config.paths.m5_features,
    )
