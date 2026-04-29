import argparse
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from agents.data_agent import get_data_range, resolve_symbol
from config.v3_config import V3Config
from core.config import load_config
from core.v3_engine import AQRSV3Engine
from strategy import execution_agent
import validator_institutional


def parse_args():
    parser = argparse.ArgumentParser(description="AQRS V3 Entrypoint")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    parser.add_argument("--mode", choices=["research", "backtest", "replay", "live", "dashboard"], default="research")
    parser.add_argument("--execute", action="store_true", help="Execute live orders when in live mode.")
    parser.add_argument("--replay-start", default=None, help="Replay start timestamp")
    parser.add_argument("--replay-end", default=None, help="Replay end timestamp")
    parser.add_argument("--replay-max-candles", type=int, default=None, help="Maximum replay candles")
    parser.add_argument("--output", default=None, help="Output CSV file path")
    parser.add_argument("--skip-readiness", action="store_true", help="Skip the final system readiness check.")
    parser.add_argument("--run-days", type=float, default=None, help="Stop live mode after this many days.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Seconds between live MT5 polling cycles.")
    parser.add_argument("--live-lookback-days", type=int, default=7, help="Rolling MT5 M5 lookback window for V3 live analysis.")
    parser.add_argument("--relaxed-demo-gate", action="store_true", help="Demo only: execute any V3 ALPHA/FLOW signal without strict SMC/session/spread gates.")
    parser.add_argument("--test-telegram", action="store_true", help="Send a Telegram test message and exit.")
    return parser.parse_args()


def get_trade_setups_path(config):
    if hasattr(config.paths, "trade_setups"):
        return config.paths.trade_setups
    if hasattr(config.paths, "features_dir"):
        return config.paths.features_dir / "trade_setups.csv"
    return Path("data/features/trade_setups.csv")


def get_research_pipeline_path(config):
    if hasattr(config.paths, "research_dir"):
        return config.paths.research_dir / "pipeline.csv"
    if hasattr(config.paths, "features_dir"):
        return config.paths.features_dir / "pipeline.csv"
    return Path("data/research/pipeline.csv")


def get_backtest_dir_path(config):
    if hasattr(config.paths, "backtest_dir"):
        return config.paths.backtest_dir
    return Path("data/backtest")


def write_output(df, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_live_m5_from_mt5(config, lookback_days):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    try:
        symbol = resolve_symbol(config.market.symbol)
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=lookback_days)
        df = get_data_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)
    finally:
        mt5.shutdown()

    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    return (
        df.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def run_research_mode(args, engine):
    result = engine.run_research(refresh_data=False)
    if args.output:
        write_output(result, args.output)
    print(f"Research complete. Generated {len(result)} rows.")
    print(result.head(10))


def run_backtest_mode(args, engine):
    result = engine.run_backtest()
    if args.output:
        write_output(result, args.output)

    alpha_count = len(result[result["signal"] == "ALPHA"])
    flow_count = len(result[result["signal"] == "FLOW"])
    print(f"\n{'=' * 40}")
    print(f"V3 BACKTEST SUMMARY ({len(result):,} candles)")
    print(f"{'=' * 40}")
    print(f"Total ALPHA Signals: {alpha_count}")
    print(f"Total FLOW Signals:  {flow_count}")
    print(f"Total Opportunities: {alpha_count + flow_count}")
    print(f"{'=' * 40}\n")


def run_replay_mode(args, engine):
    result = engine.run_replay(
        start=args.replay_start,
        end=args.replay_end,
        max_candles=args.replay_max_candles,
    )
    if args.output:
        write_output(result, args.output)
    print(f"Replay complete. Generated {len(result)} rows.")


def run_live_mode(args, config, engine):
    print(f"AQRS V3: Entering continuous LIVE monitoring for {config.market.symbol}")
    print("Press Ctrl+C to stop the system.")
    print(f"Polling every {args.poll_seconds}s with a {args.live_lookback_days}-day rolling MT5 lookback.")
    if args.relaxed_demo_gate:
        object.__setattr__(config.live, "relaxed_demo_gate", True)
        print("[DEMO] Relaxed execution gate enabled for live observation.")
    execution_agent.send_telegram_msg(config, f"AQRS V3 started\nMonitoring {config.market.symbol} 24/7...")

    last_processed_candle = None
    cycle_count = 0
    started_at = time.time()
    run_seconds = args.run_days * 24 * 60 * 60 if args.run_days else None

    while True:
        try:
            if run_seconds is not None and time.time() - started_at >= run_seconds:
                print(f"Live observation window complete after {args.run_days} day(s).")
                break

            live_m5 = load_live_m5_from_mt5(config, args.live_lookback_days)
            latest_pipeline = engine.run_research(df=live_m5)
            live_trade_setups_path = get_trade_setups_path(config)
            write_output(latest_pipeline, live_trade_setups_path)

            terminal_info = mt5.terminal_info()
            tick = mt5.symbol_info_tick(config.market.symbol)
            if tick and terminal_info:
                tick_time = pd.to_datetime(tick.time, unit="s").strftime("%H:%M:%S")
                conn_status = "CONNECTED" if terminal_info.connected else "DISCONNECTED"
                print(
                    f"[LIVE HEARTBEAT] {config.market.symbol} | Bid: {tick.bid:.5f} | "
                    f"Ask: {tick.ask:.5f} | Feed Time: {tick_time} | {conn_status}"
                )

                if not terminal_info.connected:
                    print("[WARN] Connection lost. Attempting re-initialization...")
                    mt5.initialize()
                    time.sleep(2)

            if not live_trade_setups_path.exists():
                print("[WARN] Waiting for trade setups to be generated...")
                time.sleep(10)
                continue

            live_setups_df = pd.read_csv(live_trade_setups_path, parse_dates=["time"], low_memory=False)
            if live_setups_df.empty:
                time.sleep(10)
                continue

            print(f"Market History: {len(live_setups_df)} candles loaded.")
            latest_row = live_setups_df.iloc[-1]
            current_candle_time = latest_row["time"]

            if last_processed_candle is not None and current_candle_time <= last_processed_candle:
                print(f"Waiting for next candle (Current: {current_candle_time})...")
                time.sleep(args.poll_seconds)
                continue

            print(f"\nNew candle detected: {current_candle_time}")
            print(
                latest_row[
                    [
                        "close",
                        "behavior_label",
                        "signal",
                        "confirmed_signal",
                        "quality",
                        "confirm_score",
                        "entry_price",
                        "stop_loss",
                        "take_profit",
                    ]
                ].to_string()
            )
            execution_agent.run(config, execute=args.execute)
            last_processed_candle = current_candle_time

            cycle_count += 1
            if cycle_count >= 120:
                execution_agent.send_telegram_msg(
                    config,
                    f"AQRS V3 heartbeat\nSystem active for {config.market.symbol}. No issues detected.",
                )
                cycle_count = 0

            print(f"Cycle {cycle_count} complete. Sleeping {args.poll_seconds}s...")
            time.sleep(args.poll_seconds)

        except KeyboardInterrupt:
            print("\nSystem stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] Live Cycle Error: {e}")
            time.sleep(10)


def run_readiness_check(config, engine):
    print(f"\n{'=' * 40}")
    print("FINAL SYSTEM READINESS CHECK")
    print(f"{'=' * 40}")
    try:
        research_pipeline_path = get_research_pipeline_path(config)
        if not research_pipeline_path.exists():
            print("Generating research pipeline for readiness check...")
            research_pipeline = engine.run_research(refresh_data=False)
            write_output(research_pipeline, research_pipeline_path)
            print("Research pipeline generated.")

        backtest_dir = get_backtest_dir_path(config)
        validator_input = backtest_dir / "v3_research_output.csv"
        if not validator_input.exists():
            print("Running backtest to generate data for validator...")
            bt_results = engine.run_backtest()
            write_output(bt_results, validator_input)

        validation_results = validator_institutional.main()
        readiness_score = validation_results["readiness_score"]["score"]
        readiness_tier = validation_results["readiness_score"]["tier"]

        print(f"\nOverall System Readiness Score: {readiness_score:.1f}/100 ({readiness_tier})")
        if readiness_score >= 80:
            print("[READY] System is ready to run.")
            print("\n--- Commands to run the system ---")
            print(f"1. Research: python {Path(__file__).name} --mode research --output data/research/pipeline.csv")
            print(f"2. Backtest: python {Path(__file__).name} --mode backtest --output data/backtest/v3_research_output.csv")
            print(f"3. Replay: python {Path(__file__).name} --mode replay --replay-start YYYY-MM-DD --replay-end YYYY-MM-DD --output data/replay/replay_decisions.csv")
            print(f"4. Live preview: python {Path(__file__).name} --mode live")
            print(f"5. Live execution: python {Path(__file__).name} --mode live --execute")
            print(f"6. Dashboard: python {Path(__file__).name} --mode dashboard")
        else:
            print("[NOT READY] System is not fully ready. Review the validation report.")
            print(f"Validation report: {backtest_dir / 'v3_validation_report.json'}")
    except Exception as e:
        print(f"[ERROR] An error occurred during the final system readiness check: {e}")


def main():
    args = parse_args()
    base_config = load_config(args.config)
    config = V3Config.load_from(base_config)
    engine = AQRSV3Engine(config)

    if args.test_telegram:
        result = execution_agent.test_telegram_connection(config)
        if result["ok"]:
            bot = result.get("bot", {})
            print("Telegram connected.")
            print(f"Bot: @{bot.get('username', 'unknown')}")
            print(f"Chat ID: {result.get('chat_id')}")
            print(f"Test message id: {result.get('message_id')}")
        else:
            print("Telegram connection failed.")
            print(result["reason"])
        return

    if args.mode == "research":
        run_research_mode(args, engine)
        return

    if args.mode == "backtest":
        run_backtest_mode(args, engine)
        if not args.skip_readiness:
            run_readiness_check(config, engine)
        return

    if args.mode == "replay":
        run_replay_mode(args, engine)
        return

    if args.mode == "live":
        run_live_mode(args, config, engine)
        return

    if args.mode == "dashboard":
        print("Launching AQRS V3 Standalone Dashboard...")
        subprocess.Popen(["streamlit", "run", "strategy/streamlit_app.py"])
        return


if __name__ == "__main__":
    main()
