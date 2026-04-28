import pandas as pd

from core.logging_utils import get_logger
from strategy.backtesting import _build_summary, _build_trade, _empty_trades_frame, _resolve_trade
from strategy.pipeline_transforms import run_strategy_pipeline


logger = get_logger(__name__)


def _settle_trade(active_trade, current_time, outcome, exit_price, config, equity):
    if outcome == "WIN":
        r_multiple = active_trade["reward_distance"] / active_trade["risk_distance"]
        pnl = (active_trade["risk_amount"] * r_multiple) - config.backtest.commission_per_trade
    elif outcome == "BE":
        pnl = -config.backtest.commission_per_trade
    else:
        pnl = (-active_trade["risk_amount"]) - config.backtest.commission_per_trade

    equity += pnl
    record = {
        **active_trade,
        "exit_time": current_time,
        "exit_price": exit_price,
        "result": outcome,
        "pnl": round(pnl, 2),
        "equity_after_trade": round(equity, 2),
    }
    return record, equity


def _filter_replay_frame(df, start=None, end=None, max_candles=None):
    out = df.copy()
    min_time = df["time"].min()
    max_time = df["time"].max()
    if start is not None:
        start_ts = pd.to_datetime(start)
        if isinstance(start, str) and len(start.strip()) == 10:
            start_ts = start_ts.normalize()
        start_ts = max(start_ts, min_time)
        out = out[out["time"] >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        if isinstance(end, str) and len(end.strip()) == 10:
            end_ts = end_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        end_ts = min(end_ts, max_time)
        out = out[out["time"] <= end_ts]
    out = out.sort_values("time").reset_index(drop=True)
    if max_candles is not None:
        out = out.tail(max_candles).reset_index(drop=True)
    return out


def run_replay_frame(
    replay_m5: pd.DataFrame,
    h1: pd.DataFrame,
    config,
    label: str = "replay",
):
    replay_m5 = replay_m5.sort_values("time").reset_index(drop=True)
    if replay_m5.empty:
        raise RuntimeError("Replay window is empty. Adjust replay start/end settings.")
    h1 = h1.sort_values("time").reset_index(drop=True)
    precomputed_trade_setups = run_strategy_pipeline(
        replay_m5,
        h1[h1["time"] <= replay_m5["time"].max()].copy(),
        config,
    ).reset_index(drop=True)

    decision_records = []
    event_records = []
    trade_records = []

    equity = config.backtest.starting_balance
    peak_equity = equity
    max_drawdown_pct = 0.0
    active_trades = []
    skipped_overlap = 0
    trade_id = 0

    for idx, candle in replay_m5.iterrows():
        current_time = candle["time"]

        next_active_trades = []
        resolved_this_candle = 0
        opened_this_candle = 0

        for active_trade in active_trades:
            outcome, exit_price, _ = _resolve_trade(active_trade, candle)
            if outcome:
                trade_record, equity = _settle_trade(
                    active_trade=active_trade,
                    current_time=current_time,
                    outcome=outcome,
                    exit_price=exit_price,
                    config=config,
                    equity=equity,
                )
                trade_records.append(trade_record)
                resolved_this_candle += 1
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
                event_records.append(
                    {
                        "time": current_time,
                        "event": "TRADE_CLOSED",
                        "trade_id": active_trade["trade_id"],
                        "side": active_trade["side"],
                        "setup": active_trade["setup"],
                        "quality": active_trade["quality"],
                        "market_state": active_trade["market_state"],
                        "decision": outcome,
                        "price": exit_price,
                        "pnl": trade_record["pnl"],
                        "equity_after_event": trade_record["equity_after_trade"],
                    }
                )
            else:
                next_active_trades.append(active_trade)

        active_trades = next_active_trades
        latest = precomputed_trade_setups.iloc[idx]

        action = "NO_SIGNAL"
        action_reason = "no confirmed trade setup"

        if active_trades and not config.backtest.allow_overlapping_positions:
            if latest["confirmed_signal"] in {"buy", "sell"}:
                skipped_overlap += 1
                action = "SKIP_OVERLAP"
                action_reason = "active trade already open"
                event_records.append(
                    {
                        "time": current_time,
                        "event": "SIGNAL_SKIPPED",
                        "trade_id": None,
                        "side": latest["confirmed_signal"],
                        "setup": latest.get("setup", "NONE"),
                        "quality": latest.get("quality", "NO_TRADE"),
                        "market_state": latest.get("market_state", "UNKNOWN"),
                        "decision": action,
                        "price": None,
                        "pnl": 0.0,
                        "equity_after_event": round(equity, 2),
                    }
                )
        elif latest["confirmed_signal"] in {"buy", "sell"}:
            candidate_trade = _build_trade(latest, config, equity)
            if candidate_trade is not None:
                trade_id += 1
                candidate_trade["trade_id"] = trade_id
                candidate_trade["entry_time"] = current_time
                active_trades.append(candidate_trade)
                opened_this_candle += 1
                action = f"OPEN_{latest['confirmed_signal'].upper()}"
                action_reason = "confirmed setup opened in replay"
                event_records.append(
                    {
                        "time": current_time,
                        "event": "TRADE_OPENED",
                        "trade_id": trade_id,
                        "side": candidate_trade["side"],
                        "setup": candidate_trade["setup"],
                        "quality": candidate_trade["quality"],
                        "market_state": candidate_trade["market_state"],
                        "decision": action,
                        "price": candidate_trade["executed_entry"],
                        "pnl": 0.0,
                        "equity_after_event": round(equity, 2),
                    }
                )
            else:
                action = "SKIP_INVALID_TRADE"
                action_reason = "risk or reward distance was invalid"

        # Start with all intelligence from the pipeline (which includes OHLC), then add replay-specific metadata
        decision_entry = latest.to_dict()
        decision_entry.update({
            "m5_bars_seen": idx + 1,
            "h1_bars_seen": int((h1["time"] <= current_time).sum()), # Count H1 bars up to current M5 time
            "action": action,
            "action_reason": action_reason,
            "resolved_this_candle": resolved_this_candle,
            "opened_this_candle": opened_this_candle,
            "active_trades_after_decision": len(active_trades),
            "equity_after_decision": round(equity, 2),
        })
        decision_records.append(decision_entry)

    for active_trade in active_trades:
        trade_records.append(
            {
                **active_trade,
                "exit_time": pd.NaT,
                "exit_price": None,
                "result": "OPEN",
                "pnl": 0.0,
                "equity_after_trade": round(equity, 2),
            }
        )
        event_records.append(
            {
                "time": replay_m5.iloc[-1]["time"],
                "event": "TRADE_LEFT_OPEN",
                "trade_id": active_trade["trade_id"],
                "side": active_trade["side"],
                "setup": active_trade["setup"],
                "quality": active_trade["quality"],
                "market_state": active_trade["market_state"],
                "decision": "OPEN",
                "price": None,
                "pnl": 0.0,
                "equity_after_event": round(equity, 2),
            }
        )

    decisions_df = pd.DataFrame(decision_records)
    events_df = pd.DataFrame(event_records)
    trades_df = pd.DataFrame(trade_records) if trade_records else _empty_trades_frame()

    summary = _build_summary(
        trades_df=trades_df,
        config=config,
        ending_balance=equity,
        max_drawdown_pct=max_drawdown_pct,
        skipped_overlap=skipped_overlap,
        label=label,
    )
    summary["candles_processed"] = len(replay_m5)
    summary["decision_rows"] = len(decisions_df)
    summary["events_logged"] = len(events_df)
    summary["trades_opened"] = int((events_df["event"] == "TRADE_OPENED").sum()) if not events_df.empty else 0
    summary["replay_start"] = replay_m5["time"].min()
    summary["replay_end"] = replay_m5["time"].max()
    return {
        "decisions": decisions_df,
        "events": events_df,
        "trades": trades_df,
        "summary": summary,
    }


def run(config, start=None, end=None, max_candles=None):
    m5 = pd.read_csv(config.paths.clean_m5, parse_dates=["time"])
    h1 = pd.read_csv(config.paths.clean_h1, parse_dates=["time"])

    replay_m5 = _filter_replay_frame(m5, start=start, end=end, max_candles=max_candles)
    result = run_replay_frame(replay_m5=replay_m5, h1=h1, config=config, label="replay")

    result["decisions"].to_csv(config.paths.replay_decisions, index=False)
    result["events"].to_csv(config.paths.replay_events, index=False)
    result["trades"].to_csv(config.paths.replay_trades, index=False)
    result["summary"].to_csv(config.paths.replay_summary, index=False)

    logger.info("Replay decisions saved at %s", config.paths.replay_decisions)
    logger.info("Replay events saved at %s", config.paths.replay_events)
    logger.info("Replay trades saved at %s", config.paths.replay_trades)
    logger.info("Replay summary saved at %s", config.paths.replay_summary)
    return result["decisions"]
