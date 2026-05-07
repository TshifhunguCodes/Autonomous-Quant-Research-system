import pandas as pd

from core.logging_utils import get_logger
from strategy.decision_framework import (
    get_regime_behavior,
    score_trade,
    select_strategy_mode,
    should_enter_continuation_trade,
    should_enter_counter_trend_trade,
    should_enter_retracement_trade,
    should_enter_reversal_trade,
)
from strategy.pipeline_transforms import build_signals


logger = get_logger(__name__)


def run(config):
    df = pd.read_csv(config.paths.regime_context, parse_dates=["time"])
    signals = build_signals(df)
    records = signals.to_dict(orient="records")
    signals["continuation_decision"] = [should_enter_continuation_trade(row) for row in records]
    signals["retracement_decision"] = [should_enter_retracement_trade(row) for row in records]
    signals["reversal_decision"] = [should_enter_reversal_trade(row) for row in records]
    signals["counter_trend_decision"] = [should_enter_counter_trend_trade(row) for row in records]
    signals["strategy_mode"] = [select_strategy_mode(row) for row in records]
    signals["regime_behavior"] = [get_regime_behavior(row) for row in records]
    signals["institutional_trade_score"] = [score_trade(row) for row in records]
    signals.to_csv(config.paths.signals, index=False)
    logger.info("Signal stage saved at %s", config.paths.signals)
