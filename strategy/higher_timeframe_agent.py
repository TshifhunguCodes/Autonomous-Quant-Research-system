import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import merge_h1_context_into_m5, build_simple_htf_bias


logger = get_logger(__name__)
def run(config):
    m5 = pd.read_csv(config.paths.m5_features, parse_dates=["time"])
    h1 = pd.read_csv(config.paths.clean_h1, parse_dates=["time"])
    h4 = pd.read_csv(config.paths.clean_h4, parse_dates=["time"])
    d1 = pd.read_csv(config.paths.clean_d1, parse_dates=["time"])

    # Step 1: Merge existing H1 context
    m5 = merge_h1_context_into_m5(m5, h1)

    # Step 2: Merge simple bias for H4 and D1
    for tf_df, prefix in [(h4, "h4"), (d1, "d1")]:
        bias_df = build_simple_htf_bias(tf_df, prefix)
        m5 = pd.merge_asof(
            m5.sort_values("time"),
            bias_df.sort_values("time"),
            on="time",
            direction="backward"
        )

    m5.to_csv(config.paths.m5_features, index=False)
    logger.info(
        "Multi-timeframe context merged into %s",
        config.paths.m5_features,
    )
