import argparse
from dataclasses import replace

from core.config import ensure_runtime_dirs, load_config
from core.logging_utils import configure_logging, get_logger
from core.pipeline import run_backtest, run_live, run_replay, run_research, run_stress, run_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Quant Research System V2"
    )
    parser.add_argument(
        "--mode",
        choices=["research", "backtest", "live", "summary", "replay", "stress"],
        default="research",
        help="Select which pipeline mode to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a JSON config file.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch fresh MT5 data before running the pipeline.",
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Reuse existing research artifacts instead of rebuilding them.",
    )
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Actually send a live order in live mode. Without this flag, live mode is preview-only.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override the configured log level for this run.",
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=7,
        help="Window size in days for rolling backtest reports.",
    )
    parser.add_argument(
        "--rolling-step-days",
        type=int,
        default=7,
        help="Step size in days between rolling backtest windows.",
    )
    parser.add_argument(
        "--replay-start",
        default=None,
        help="Optional replay start date/time, e.g. 2026-04-01 or 2026-04-01 08:00:00.",
    )
    parser.add_argument(
        "--replay-end",
        default=None,
        help="Optional replay end date/time, e.g. 2026-04-22 or 2026-04-22 12:25:00.",
    )
    parser.add_argument(
        "--replay-max-candles",
        type=int,
        default=None,
        help="Optional cap on replay candles for faster debugging runs.",
    )
    parser.add_argument(
        "--stress-random-runs",
        type=int,
        default=6,
        help="Number of random walk-forward slices to evaluate in stress mode.",
    )
    parser.add_argument(
        "--stress-window-days",
        type=int,
        default=5,
        help="Window size in days for each random walk-forward stress slice.",
    )
    parser.add_argument(
        "--stress-seed",
        type=int,
        default=42,
        help="Random seed for reproducible stress-test slice selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_runtime_dirs(config)
    effective_log_level = args.log_level or config.logging.level
    if args.mode == "summary" and args.log_level is None:
        effective_log_level = "ERROR"
    configure_logging(config.paths.app_log, effective_log_level)
    logger = get_logger(__name__)

    logger.info("Running mode=%s", args.mode)

    if args.mode == "research":
        run_research(config, refresh_data=args.refresh_data)
        return

    if args.mode == "backtest":
        run_backtest(
            config,
            refresh_data=args.refresh_data,
            reuse_artifacts=args.reuse_artifacts,
            rolling_window_days=args.rolling_window_days,
            rolling_step_days=args.rolling_step_days,
        )
        return

    if args.mode == "summary":
        run_summary(
            config,
            refresh_data=args.refresh_data,
            reuse_artifacts=args.reuse_artifacts,
            rolling_window_days=args.rolling_window_days,
            rolling_step_days=args.rolling_step_days,
        )
        return

    if args.mode == "replay":
        run_replay(
            config,
            refresh_data=args.refresh_data,
            replay_start=args.replay_start,
            replay_end=args.replay_end,
            replay_max_candles=args.replay_max_candles,
        )
        return

    if args.mode == "stress":
        run_stress(
            config,
            refresh_data=args.refresh_data,
            stress_random_runs=args.stress_random_runs,
            stress_window_days=args.stress_window_days,
            stress_seed=args.stress_seed,
        )
        return

    # Multi-Symbol Refactor: Support portfolio-level iteration
    portfolio = getattr(config, "portfolio", [config.market.symbol])
    for symbol in portfolio:
        # Create a new config context for this symbol because the dataclass is frozen
        symbol_config = replace(config, market=replace(config.market, symbol=symbol))

        logger.info("Processing symbol=%s in live mode", symbol)
        run_live(
            symbol_config,
            refresh_data=args.refresh_data,
            reuse_artifacts=args.reuse_artifacts,
            execute=args.execute_live,
        )


if __name__ == "__main__":
    main()
