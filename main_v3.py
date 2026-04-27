import argparse
import subprocess
import pandas as pd
from pathlib import Path
import numpy as np # For float comparison tolerance

from core.config import load_config
from core.v3_engine import AQRSV3Engine
from config.v3_config import V3Config
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
        print("Running in LIVE mode...")
        
        # --- Consistency Check: Live Signal vs Research Artifacts ---
        live_trade_setups_path = get_trade_setups_path(config)
        research_pipeline_path = get_research_pipeline_path(config)

        if not live_trade_setups_path.exists():
            print("[WARN] No live trade setups found. Cannot perform consistency check.")
            # Proceed with live execution if --execute is present, or just exit preview
            if args.execute:
                print("Executing live trades without consistency check due to missing live setups.")
                engine.run_live(execute=True) # Assuming this method exists and handles execution
            else:
                print("Live preview mode. No execution.")
            return

        if not research_pipeline_path.exists():
            print(f"[WARN] Research pipeline artifact not found at {research_pipeline_path}. Cannot perform consistency check.")
            # Proceed with live execution if --execute is present, or just exit preview
            if args.execute:
                print("Executing live trades without consistency check due to missing research artifact.")
                engine.run_live(execute=True)
            else:
                print("Live preview mode. No execution.")
            return

        try:
            live_setups_df = pd.read_csv(live_trade_setups_path, parse_dates=["time"], low_memory=False)
            research_df = pd.read_csv(research_pipeline_path, parse_dates=["time"], low_memory=False)

            if live_setups_df.empty:
                print("[WARN] Live trade setups file is empty. Cannot perform consistency check.")
                if args.execute:
                    print("Executing live trades without consistency check due to empty live setups.")
                    engine.run_live(execute=True)
                else:
                    print("Live preview mode. No execution.")
                return

            latest_live_signal = live_setups_df.iloc[-1]
            
            # Find the corresponding signal in the research pipeline
            # Match by time and confirmed_signal (or other unique identifiers)
            matching_research_signals = research_df[
                (research_df["time"] == latest_live_signal["time"]) &
                (research_df["confirmed_signal"] == latest_live_signal["confirmed_signal"])
            ]

            if matching_research_signals.empty:
                print(f"[FAIL] Consistency Check Failed: No matching research artifact found for live signal at {latest_live_signal['time']}.")
                print(f"   Live Signal: {latest_live_signal['confirmed_signal']} @ {latest_live_signal['entry_price']:.5f}")
                if args.execute:
                    print("Blocking live execution due to missing research comparison.")
                    return
            else:
                # Assuming the first match is the correct one, or there's only one
                matching_research_signal = matching_research_signals.iloc[0]

                discrepancies = []
                fields_to_check = ["entry_price", "stop_loss", "take_profit"]
                tolerance = 0.0001 # Points tolerance for price comparison

                for field in fields_to_check:
                    live_val = latest_live_signal.get(field)
                    research_val = matching_research_signal.get(field)

                    if pd.isna(live_val) and pd.isna(research_val):
                        continue # Both are NaN, considered consistent
                    if pd.isna(live_val) or pd.isna(research_val):
                        discrepancies.append(f"  - {field}: Live={live_val}, Research={research_val} (NaN mismatch)")
                        continue
                    
                    # Convert to float for comparison, handle potential non-numeric types gracefully
                    try:
                        live_val_f = float(live_val)
                        research_val_f = float(research_val)
                        if not np.isclose(live_val_f, research_val_f, atol=tolerance):
                            discrepancies.append(f"  - {field}: Live={live_val_f:.5f}, Research={research_val_f:.5f} (Difference > {tolerance})")
                    except ValueError:
                        discrepancies.append(f"  - {field}: Non-numeric value encountered (Live={live_val}, Research={research_val})")


                if discrepancies:
                    print(f"[FAIL] Consistency Check Failed for live signal at {latest_live_signal['time']}:")
                    for d in discrepancies:
                        print(d)
                    if args.execute:
                        print("Blocking live execution due to inconsistencies with research artifacts.")
                        return
                else:
                    print(f"[PASS] Consistency Check Passed for live signal at {latest_live_signal['time']}.")
                    print(f"   Live Signal: {latest_live_signal['confirmed_signal']} @ {latest_live_signal['entry_price']:.5f}")
                    print(f"   SL: {latest_live_signal['stop_loss']:.5f}, TP: {latest_live_signal['take_profit']:.5f}")

            # If consistency check passes (or not blocking), proceed with live execution if requested
            if args.execute:
                print("Proceeding with live execution...")
                engine.run_live(execute=True) # Assuming this method exists and handles execution
            else:
                print("Live preview mode. No execution.")

        except Exception as e:
            print(f"[ERROR] An error occurred during live consistency check: {e}")
            if args.execute:
                print("Blocking live execution due to error in consistency check.")
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
