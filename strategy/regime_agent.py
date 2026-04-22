import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_regime_layer


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.market_state, parse_dates=["time"])
    build_regime_layer(df, config).to_csv(config.paths.regime_context, index=False)
    logger.info("Regime stage saved at %s", config.paths.regime_context)
