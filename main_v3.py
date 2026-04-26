import argparse

from core.config import load_config
from core.v3_engine import AQRSV3Engine
from config.v3_config import V3Config


def parse_args():
    parser = argparse.ArgumentParser(description="AQRS V3 Entrypoint")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    parser.add_argument("--mode", choices=["research", "backtest", "replay", "live", "dashboard"], default="research")
    parser.add_argument("--execute", action="store_true", help="Execute live orders when in live mode.")
    parser.add_argument("--replay-start", default=None, help="Replay start timestamp")
    parser.add_argument("--replay-end", default=None, help="Replay end timestamp")
    parser.add_argument("--replay-max-candles", type=int, default=None, help="Maximum replay candles")
    parser.add_argument("--output", default=None, help="Output CSV file path")
    return parser.parse_args()


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
        return

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
        return

    if args.mode == "replay":
        result = engine.run_replay(start=args.replay_start, end=args.replay_end, max_candles=args.replay_max_candles)
        if args.output:
            result.to_csv(args.output, index=False)
        print(f"Replay complete. Generated {len(result)} rows.")
        return

    if args.mode == "live":
        print("Live mode not yet implemented.")
        return

    if args.mode == "dashboard":
        print("Dashboard mode not yet implemented.")
        return


if __name__ == "__main__":
    main()
