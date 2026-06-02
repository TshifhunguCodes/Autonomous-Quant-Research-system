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
from strategy.backtesting import run_backtest_frame
from core.logging_utils import get_logger

logger = get_logger(__name__)
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
    # Persistent connection assumed; initialization handled in run_live_mode
    symbol = resolve_symbol(config.market.symbol)
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=lookback_days)
    df = get_data_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)

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
    trades_df, execution_summary = run_backtest_frame(result, engine.config, label="COMBINED", mode="COMBINED")
    backtest_dir = get_backtest_dir_path(engine.config)
    backtest_dir.mkdir(parents=True, exist_ok=True)
    trades_path = backtest_dir / "v3_executed_trades.csv"
    summary_path = backtest_dir / "v3_execution_summary.csv"
    session_path = backtest_dir / "v3_session_performance.csv"
    trades_df.to_csv(trades_path, index=False)
    execution_summary.to_csv(summary_path, index=False)
    session_summary = _build_session_summary(trades_df)
    session_summary.to_csv(session_path, index=False)

    print(f"\n{'=' * 40}")
    print(f"V3 BACKTEST SUMMARY ({len(result):,} candles)")
    print(f"{'=' * 40}")
    print(f"Total ALPHA Signals: {alpha_count}")
    print(f"Total FLOW Signals:  {flow_count}")
    print(f"Total Opportunities: {alpha_count + flow_count}")
    print("-" * 40)
    if execution_summary.empty:
        print("No executed trade summary generated.")
    else:
        row = execution_summary.iloc[0]
        print(f"Trades Taken:       {int(row.get('closed_trades', 0))}")
        print(f"Wins:               {int(row.get('wins', 0))}")
        print(f"Losses:             {int(row.get('losses', 0))}")
        print(f"Open Trades:        {int(row.get('open_trades', 0))}")
        print(f"Win Rate:           {float(row.get('true_win_rate_pct', 0.0)):.2f}%")
        print(f"Profit Factor:      {float(row.get('profit_factor', 0.0)):.2f}")
        print(f"Net PnL:            {float(row.get('net_pnl', 0.0)):.2f}")
        print(f"Max Drawdown:       {float(row.get('max_drawdown_pct', 0.0)):.2f}%")
        print(f"Trades Per Day:     {float(row.get('trades_per_day', 0.0)):.2f}")
    if not session_summary.empty:
        print("-" * 40)
        print("SESSION PERFORMANCE")
        print(session_summary.to_string(index=False))
    print("-" * 40)
    print(f"Executed trades: {trades_path}")
    print(f"Summary CSV:     {summary_path}")
    print(f"Session CSV:     {session_path}")
    print(f"{'=' * 40}\n")


def _build_session_summary(trades_df):
    if trades_df.empty or "session" not in trades_df.columns:
        return pd.DataFrame(
            columns=["session", "trades", "wins", "losses", "breakevens", "win_rate_pct", "net_pnl", "profit_factor"]
        )
    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS", "BE"])].copy()
    if closed.empty:
        return pd.DataFrame(
            columns=["session", "trades", "wins", "losses", "breakevens", "win_rate_pct", "net_pnl", "profit_factor"]
        )

    rows = []
    for session, group in closed.groupby("session", dropna=False):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        breakevens = int((group["result"] == "BE").sum())
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(abs(pnl[pnl < 0].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
        decided = wins + losses
        rows.append(
            {
                "session": session,
                "trades": int(len(group)),
                "wins": wins,
                "losses": losses,
                "breakevens": breakevens,
                "win_rate_pct": round((wins / decided * 100.0) if decided else 0.0, 2),
                "net_pnl": round(float(pnl.sum()), 2),
                "profit_factor": round(profit_factor, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["net_pnl", "trades"], ascending=[False, False])


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
    
    if not mt5.initialize():
        logger.critical("Failed to initialize MT5 for live mode.")
        return

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

            start_load_time = time.time()
            live_m5 = load_live_m5_from_mt5(config, args.live_lookback_days)
            load_duration = time.time() - start_load_time
            logger.info(f"MT5 data loaded in {load_duration:.2f}s ({len(live_m5)} candles).")

            start_research_time = time.time()
            latest_pipeline = engine.run_research(df=live_m5)
            research_duration = time.time() - start_research_time
            logger.info(f"Research pipeline processed in {research_duration:.2f}s.")
            live_trade_setups_path = get_trade_setups_path(config)
            write_output(latest_pipeline, live_trade_setups_path)

            terminal_info = mt5.terminal_info()
            tick = mt5.symbol_info_tick(config.market.symbol)
            if tick and terminal_info:
                broker_dt = pd.to_datetime(tick.time, unit="s")
                local_dt = datetime.now()
                
                # Calculate the hour offset for the heartbeat display
                raw_diff_seconds = (local_dt - broker_dt).total_seconds()
                hour_offset = round(raw_diff_seconds / 3600)
                
                conn_status = "CONNECTED" if terminal_info.connected else "DISCONNECTED"
                print(
                    f"[LIVE HEARTBEAT] {config.market.symbol} | Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f} | "
                    f"Broker Time: {broker_dt.strftime('%H:%M:%S')} | Local Time: {local_dt.strftime('%H:%M:%S')} | "
                    f"Offset: {hour_offset:+.0f}h | {conn_status}"
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
            # Ensure consistent datetime precision
            if "time" in live_setups_df.columns:
                live_setups_df["time"] = live_setups_df["time"].astype("datetime64[s]")
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
            # Optimization: Pass the latest row directly to avoid redundant disk reads
            execution_agent.run(config, execute=args.execute, signal_data=latest_row.to_dict())
            last_processed_candle = current_candle_time

            # ===== AUTO-RETRAIN ML MODELS =====
            try:
                retrain_result = engine.retrain_pipeline.check_and_retrain(latest_pipeline)
                if retrain_result.get("retrained"):
                    logger.info(f"🤖 ML models auto-retrained: {retrain_result.get('total_retrains')} total retrains")
            except Exception as e:
                logger.warning(f"Auto-retrain check failed (non-blocking): {e}")
            # ===================================

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
            mt5.shutdown()
            break
        except Exception as e:
            print(f"[ERROR] Live Cycle Error: {e}")
            # Don't shutdown MT5 on error - try to recover
            logger.error("Attempting to recover MT5 connection...")
            try:
                if not mt5.initialize():
                    logger.error("Failed to reinitialize MT5")
            except:
                pass
            time.sleep(10)
            continue


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
