import argparse
import MetaTrader5 as mt5
import subprocess
import pandas as pd
from pathlib import Path
import numpy as np # For float comparison tolerance
import time

from core.config import load_config
from core.v3_engine import AQRSV3Engine
from config.v3_config import V3Config
from strategy import execution_agent
import validator_institutional # Import the validator

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
    return parser.parse_args()

def get_trade_setups_path(config):
    """Safe helper to get the trade setups path even if the attribute is missing."""
    if hasattr(config.paths, "trade_setups"):
        return config.paths.trade_setups
    # Fallback to standard V3 features directory
    fallback = Path("data/features/trade_setups.csv")
    if hasattr(config.paths, "features_dir"):
        return config.paths.features_dir / "trade_setups.csv"
    return fallback

def get_research_pipeline_path(config):
    """Safe helper to get the research pipeline path even if the attribute is missing."""
    if hasattr(config.paths, "research_dir"):
        return config.paths.research_dir / "pipeline.csv"
    # Fallback to standard V3 research directory
    return Path("data/research/pipeline.csv")

def get_backtest_dir_path(config):
    """Safe helper to get the backtest directory path even if the attribute is missing."""
    if hasattr(config.paths, "backtest_dir"):
        return config.paths.backtest_dir
    # Fallback to standard V3 backtest directory
    return Path("data/backtest")


def main():
    args = parse_args()
    base_config = load_config(args.config)
    config = V3Config.load_from(base_config)
    engine = AQRSV3Engine(config)

    if args.mode == "research":
        result = engine.run_research(refresh_data=False)
        if args.output:
            result.to_csv(args.output, index=False)
        print(f"Research complete. Generated {len(result)} rows.")
        print(result.head(10))

    if args.mode == "backtest":
        result = engine.run_backtest()
        if args.output:
            result.to_csv(args.output, index=False)
        
        # Calculate and print trade summary
        alpha_count = len(result[result['signal'] == 'ALPHA'])
        flow_count = len(result[result['signal'] == 'FLOW'])
        print(f"\n{'='*40}")
        print(f"V3 BACKTEST SUMMARY ({len(result):,} candles)")
        print(f"{'='*40}")
        print(f"Total ALPHA Signals: {alpha_count}")
        print(f"Total FLOW Signals:  {flow_count}")
        print(f"Total Opportunities: {alpha_count + flow_count}")
        print(f"{'='*40}\n")

    if args.mode == "replay":
        result = engine.run_replay(start=args.replay_start, end=args.replay_end, max_candles=args.replay_max_candles)
        if args.output:
            result.to_csv(args.output, index=False)
        print(f"Replay complete. Generated {len(result)} rows.")

    if args.mode == "live":
        print(f"🚀 AQRS V3: Entering Continuous LIVE monitoring for {config.market.symbol}")
        print("Press Ctrl+C to stop the system.")
        
        # Notify startup
        execution_agent.send_telegram_msg(config, f"🟢 **AQRS V3 Started**\nMonitoring {config.market.symbol} 24/7...")

        last_processed_candle = None
        cycle_count = 0

        while True:
            try:
                # 1. Force the engine to fetch latest market data and recalculate V3 signals
                # refresh_data=True tells the engine to pull new bars from MT5
                engine.run_research(refresh_data=True)
                
                # --- Live Feed Verification Heartbeat ---
                tick = mt5.symbol_info_tick(config.market.symbol)
                if tick:
                    print(f"📡 [LIVE HEARTBEAT] {config.market.symbol} | Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f}")

                # --- Consistency Check: Live Signal vs Research Artifacts ---
                live_trade_setups_path = get_trade_setups_path(config)
                research_pipeline_path = get_research_pipeline_path(config)

                if not live_trade_setups_path.exists():
                    print("[WARN] Waiting for trade setups to be generated...")
                    time.sleep(10)
                    continue

                live_setups_df = pd.read_csv(live_trade_setups_path, parse_dates=["time"], low_memory=False)
                if live_setups_df.empty:
                    time.sleep(10)
                    continue
                
                latest_row = live_setups_df.iloc[-1]
                current_candle_time = latest_row["time"]

                # Only process if we have a brand new candle
                if last_processed_candle is not None and current_candle_time <= last_processed_candle:
                    time.sleep(10)
                    continue
                
                print(f"\n🔔 New Candle Detected: {current_candle_time}")
                
                # Run the execution agent (this handles the Execution Gate and the mobile alert)
                execution_agent.run(config, execute=args.execute)
                
                last_processed_candle = current_candle_time
                
                # Heartbeat every 120 cycles (approx every 1 hour)
                cycle_count += 1
                if cycle_count >= 120:
                    execution_agent.send_telegram_msg(config, f"🛡️ **AQRS V3 Heartbeat**\nSystem active for {config.market.symbol}. No issues detected.")
                    cycle_count = 0

                # Polling interval (wait for 30 seconds before checking for new data again)
                print(f"💤 Cycle {cycle_count} complete. Sleeping 30s...")
                time.sleep(30)

            except KeyboardInterrupt:
                print("\n🛑 System stopped by user.")
                break
            except Exception as e:
                print(f"[ERROR] Live Cycle Error: {e}")
                time.sleep(10)
        return

    if args.mode == "dashboard":
        print("🚀 Launching AQRS V3 Standalone Dashboard...")
        subprocess.Popen(["streamlit", "run", "strategy/streamlit_app.py"])
        return

    # Skip readiness check if explicitly requested or in high-performance modes
    if args.skip_readiness or args.mode in ["research", "replay"]:
        return

    # --- Final System Readiness Check ---
    print(f"\n{'='*40}")
    print("FINAL SYSTEM READINESS CHECK")
    print(f"{'='*40}")
    try:
        # Ensure the research pipeline is generated before running the validator
        research_pipeline_path = get_research_pipeline_path(config)
        if not research_pipeline_path.exists():
            print("Generating research pipeline for readiness check...")
            engine.run_research(refresh_data=False)
            print("Research pipeline generated.")
        
        # Ensure v3_research_output.csv exists for the validator
        backtest_dir = get_backtest_dir_path(config)
        validator_input = backtest_dir / "v3_research_output.csv"
        
        if not validator_input.exists():
            print("Running backtest to generate data for validator...")
            bt_results = engine.run_backtest()
            bt_results.to_csv(validator_input, index=False)

        validation_results = validator_institutional.main()
        readiness_score = validation_results["readiness_score"]["score"]
        readiness_tier = validation_results["readiness_score"]["tier"]
        
        print(f"\nOverall System Readiness Score: {readiness_score:.1f}/100 ({readiness_tier})")
        if readiness_score >= 80:
            print("[READY] System is ready to be ran!")
            print("\n--- Commands to run the system ---")
            print("1. To run in Research mode (generate fresh pipeline data):")
            print(f"   python {Path(__file__).name} --mode research --output data/research/pipeline.csv")
            print("2. To run a full Backtest (generate performance reports):")
            print(f"   python {Path(__file__).name} --mode backtest --output data/backtest/v3_research_output.csv")
            print("3. To run in Replay mode (simulate candle-by-candle):")
            print(f"   python {Path(__file__).name} --mode replay --replay-start YYYY-MM-DD --replay-end YYYY-MM-DD --output data/replay/replay_decisions.csv")
            print("4. To run in Live Preview mode (check current signals without execution):")
            print(f"   python {Path(__file__).name} --mode live")
            print("5. To run in Live Execution mode (send orders to MT5):")
            print(f"   python {Path(__file__).name} --mode live --execute")
            print("6. To launch the Standalone Dashboard:")
            print(f"   python {Path(__file__).name} --mode dashboard")
        else:
            print("[NOT READY] System is NOT fully ready. Please review the validation report for details.")
            print(f"   Validation report: {backtest_dir / 'v3_validation_report.json'}")

    except Exception as e:
        print(f"[ERROR] An error occurred during the final system readiness check: {e}")


if __name__ == "__main__":
    main()
