from core.logging_utils import get_logger

from agents.cleaning_agent import run as clean_run
from agents.data_agent import run as data_run
from agents.feature_agent import run as feature_run
from strategy.backtesting import run as backtesting_run
from strategy.reporting import run as reporting_run
from strategy.replay_engine import run as replay_run
from strategy.stress_testing import run as stress_run
from strategy.confirmation_agent import run as confirmation_run
from strategy.debug_engine import run as debug_run
from strategy.entry_agent import run as entry_run
from strategy.execution_agent import run as execution_run
from strategy.higher_timeframe_agent import run as higher_timeframe_run
from strategy.market_state_agent import run as market_state_run
from strategy.regime_agent import run as regime_run
from strategy.setup_engine import run as setup_run
from strategy.signal_agent import run as signal_run
from strategy.structure_agent import run as structure_run
from strategy.zone_agent import run as zone_run


logger = get_logger(__name__)


def _raw_data_available(config) -> bool:
    return config.paths.raw_m5.exists() and config.paths.raw_h1.exists()


def _clean_data_available(config) -> bool:
    return config.paths.clean_m5.exists() and config.paths.clean_h1.exists()


def _research_artifacts_available(config) -> bool:
    return config.paths.trade_setups.exists()


def _backtest_reports_available(config) -> bool:
    return (
        config.paths.backtest_summary.exists()
        and config.paths.session_performance.exists()
        and config.paths.best_setup_types.exists()
    )


def run_research(config, refresh_data: bool = False):
    logger.info("Starting research mode")

    if refresh_data or not _raw_data_available(config):
        logger.info("Fetching fresh MT5 data")
        data_run(config)
    else:
        logger.info("Using cached raw data")

    clean_run(config)
    feature_run(config)
    higher_timeframe_run(config)
    structure_run(config)
    zone_run(config)
    market_state_run(config)
    regime_run(config)
    signal_run(config)
    setup_run(config)
    confirmation_run(config)
    entry_run(config)

    logger.info("Research artifacts ready at %s", config.paths.trade_setups)
    return config.paths.trade_setups


def run_backtest(
    config,
    refresh_data: bool = False,
    reuse_artifacts: bool = False,
    rolling_window_days: int = 7,
    rolling_step_days: int = 7,
    in_sample_end: str | None = None,
    oos_start: str | None = None,
):
    logger.info("Starting backtest mode")

    if refresh_data or not reuse_artifacts or not _research_artifacts_available(config):
        run_research(config, refresh_data=refresh_data)
    else:
        logger.info("Reusing existing research artifacts")

    backtesting_run(config, in_sample_end=in_sample_end, oos_start=oos_start)
    reporting_run(
        config,
        rolling_window_days=rolling_window_days,
        rolling_step_days=rolling_step_days,
    )
    debug_run(config)
    logger.info("Backtest outputs ready at %s", config.paths.backtest_summary)
    return config.paths.backtest_summary


def run_live(
    config,
    refresh_data: bool = False,
    reuse_artifacts: bool = False,
    execute: bool = False,
):
    logger.info("Starting live mode")

    if refresh_data or not reuse_artifacts or not _research_artifacts_available(config):
        run_research(config, refresh_data=refresh_data)
    else:
        logger.info("Reusing existing research artifacts")

    result = execution_run(config, execute=execute)
    debug_run(config)
    return result


def run_summary(
    config,
    refresh_data: bool = False,
    reuse_artifacts: bool = False,
    rolling_window_days: int = 7,
    rolling_step_days: int = 7,
    in_sample_end: str | None = None,
    oos_start: str | None = None,
):
    logger.info("Starting summary mode")

    if refresh_data or not reuse_artifacts or not _research_artifacts_available(config):
        run_research(config, refresh_data=refresh_data)
        backtesting_run(config, in_sample_end=in_sample_end, oos_start=oos_start)
        reporting_run(
            config,
            rolling_window_days=rolling_window_days,
            rolling_step_days=rolling_step_days,
        )
    elif not _backtest_reports_available(config):
        backtesting_run(config, in_sample_end=in_sample_end, oos_start=oos_start)
        reporting_run(
            config,
            rolling_window_days=rolling_window_days,
            rolling_step_days=rolling_step_days,
        )
    else:
        logger.info("Reusing existing research and backtest artifacts")

    return debug_run(config, print_terminal=True)


def run_replay(
    config,
    refresh_data: bool = False,
    replay_start: str | None = None,
    replay_end: str | None = None,
    replay_max_candles: int | None = None,
):
    logger.info("Starting replay mode")

    if refresh_data or not _raw_data_available(config):
        logger.info("Fetching fresh MT5 data")
        data_run(config)

    if refresh_data or not _clean_data_available(config):
        clean_run(config)

    return replay_run(
        config,
        start=replay_start,
        end=replay_end,
        max_candles=replay_max_candles,
    )


def run_stress(
    config,
    refresh_data: bool = False,
    stress_random_runs: int = 6,
    stress_window_days: int = 5,
    stress_seed: int = 42,
):
    logger.info("Starting stress-test mode")

    if refresh_data or not _raw_data_available(config):
        logger.info("Fetching fresh MT5 data")
        data_run(config)

    if refresh_data or not _clean_data_available(config):
        clean_run(config)

    return stress_run(
        config,
        random_runs=stress_random_runs,
        window_days=stress_window_days,
        seed=stress_seed,
    )
