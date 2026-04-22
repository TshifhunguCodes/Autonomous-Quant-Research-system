import pandas as pd

from core.logging_utils import get_logger


logger = get_logger(__name__)


def _build_trade(row, config, equity):
    if not bool(row.get("trade_allowed", True)):
        return None

    # --- HARD TRUTH FILTERS ---
    # 1. Elite Paradox: Reject exhaustion signals (> 100)
    if float(row.get("confirm_score", 0)) > 100:
        return None

    # 2. Volatility Regime Adaptation
    state = row.get("market_state", "UNKNOWN")
    quality = row.get("quality", "UNKNOWN")
    risk_adj = 1.0

    if state == "CHOPPY":  # Low Volatility: skip low-quality, require ELITE
        if quality != "ELITE":
            return None
        score_floor = 80
    elif state == "VOLATILE":  # High Volatility: ELITE only, Major Zones, 0.5x risk
        if quality != "ELITE":
            return None
        if not (bool(row.get("major_support", 0)) or bool(row.get("major_resistance", 0))):
            return None
        score_floor = 90
        risk_adj = 0.5
    else:  # Medium Volatility (RANGING, TRENDING)
        if state == "RANGING":
            if quality not in ["ELITE", "HIGH"]:
                return None
            score_floor = 65
        else:  # Default Trending
            if quality != "ELITE":
                return None
            score_floor = 80

    # 3. Timeframe & Conviction Filters
    if int(row.get("h1_alignment", 0)) != 1 or float(row.get("confirm_score", 0)) < score_floor:
        return None

    signal = row["confirmed_signal"]
    spread_price = float(row.get("spread", 0) or 0) * config.market.point_size
    slippage_price = config.backtest.slippage_points * config.market.point_size
    entry_price = float(row["entry_price"])
    stop_loss = float(row["stop_loss"])
    take_profit = float(row["take_profit"])
    if pd.isna(entry_price) or pd.isna(stop_loss) or pd.isna(take_profit):
        return None

    if signal == "buy":
        executed_entry = entry_price + spread_price + slippage_price
        risk_distance = executed_entry - stop_loss
        reward_distance = take_profit - executed_entry
    else:
        executed_entry = entry_price - spread_price - slippage_price
        risk_distance = stop_loss - executed_entry
        reward_distance = executed_entry - take_profit

    if risk_distance <= 0 or reward_distance <= 0:
        return None

    return {
        "signal_time": row["time"],
        "side": signal,
        "setup": row.get("setup", "UNKNOWN"),
        "signal_label": row.get("signal", "UNKNOWN"),
        "quality": row.get("quality", "UNKNOWN"),
        "market_state": row.get("market_state", "UNKNOWN"),
        "market_regime": row.get("market_regime", "UNKNOWN"),
        "h1_bias": row.get("h1_bias", "UNKNOWN"),
        "h1_alignment": row.get("h1_alignment", 0),
        "entry_price": entry_price,
        "executed_entry": executed_entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_distance": risk_distance,
        "reward_distance": reward_distance,
        "risk_multiplier": float(row.get("risk_multiplier", 1.0) or 1.0),
        "risk_amount": equity
        * config.backtest.risk_per_trade
        * float(row.get("risk_multiplier", 1.0) or 1.0)
        * risk_adj,
        "confirm_score": row.get("confirm_score", 0),
    }

def _resolve_trade(trade, candle):
    """
    Resolves a trade and manages Break-Even logic.
    Returns (Outcome, ExitPrice, was_be_stop)
    """
    high = float(candle["high"])
    low = float(candle["low"])
    
    # 1. Update Break-Even status
    # HIGH volatility uses 1.2R for faster locking, MEDIUM/LOW uses 1.5R for breathing room.
    if not trade.get("is_be", False):
        be_threshold = 1.2 if trade.get("market_state") == "VOLATILE" else 1.5
        
        if trade["side"] == "buy" and high >= (trade["executed_entry"] + (trade["risk_distance"] * be_threshold)):
            trade["stop_loss"] = trade["executed_entry"]
            trade["is_be"] = True
        elif trade["side"] == "sell" and low <= (trade["executed_entry"] - (trade["risk_distance"] * be_threshold)):
            trade["stop_loss"] = trade["executed_entry"]
            trade["is_be"] = True

    if trade["side"] == "buy":
        if low <= trade["stop_loss"]:
            # If is_be is true, this is a BE exit, not a full LOSS
            outcome = "BE" if trade.get("is_be", False) else "LOSS"
            return outcome, trade["stop_loss"], trade.get("is_be", False)
        if high >= trade["take_profit"]:
            return "WIN", trade["take_profit"], False
    else:
        if high >= trade["stop_loss"]:
            outcome = "BE" if trade.get("is_be", False) else "LOSS"
            return outcome, trade["stop_loss"], trade.get("is_be", False)
        if low <= trade["take_profit"]:
            return "WIN", trade["take_profit"], False

    return None, None, False


def _empty_trades_frame():
    return pd.DataFrame(
        columns=[
            "signal_time",
            "side",
            "setup",
            "signal_label",
            "quality",
            "market_state",
            "market_regime",
            "h1_bias",
            "h1_alignment",
            "entry_price",
            "executed_entry",
            "stop_loss",
            "take_profit",
            "risk_distance",
            "reward_distance",
            "risk_multiplier",
            "risk_amount",
            "confirm_score",
            "exit_time",
            "exit_price",
            "result",
            "reentry_count",
            "pnl",
            "equity_after_trade",
        ]
    )


def _build_summary(trades_df, config, ending_balance, max_drawdown_pct, skipped_overlap, label):
    closed_trades = (
        trades_df[trades_df["result"].isin(["WIN", "LOSS", "BE"])]
        if not trades_df.empty
        else pd.DataFrame()
    )
    wins = int((closed_trades["result"] == "WIN").sum()) if not closed_trades.empty else 0
    losses = int((closed_trades["result"] == "LOSS").sum()) if not closed_trades.empty else 0
    gross_profit = (
        float(closed_trades.loc[closed_trades["pnl"] > 0, "pnl"].sum())
        if not closed_trades.empty
        else 0.0
    )
    gross_loss = (
        float(closed_trades.loc[closed_trades["pnl"] < 0, "pnl"].sum())
        if not closed_trades.empty
        else 0.0
    )
    profit_factor = round(gross_profit / abs(gross_loss), 2) if gross_loss else 0.0
    win_rate = (
        round((wins / len(closed_trades)) * 100, 2)
        if len(closed_trades)
        else 0.0
    )
    # True Win Rate: Accuracy of the signal itself (excluding BE scratches)
    true_win_rate = (
        round((wins / (wins + losses)) * 100, 2)
        if (wins + losses) > 0
        else 0.0
    )

    # Calculate Sample Duration for Trade Frequency
    duration_days = 0
    if not closed_trades.empty:
        start_dt = pd.to_datetime(closed_trades["signal_time"]).min()
        end_dt = pd.to_datetime(closed_trades["exit_time"]).max()
        if pd.notna(start_dt) and pd.notna(end_dt):
            duration_days = max(1, (end_dt - start_dt).days)
    trades_per_day = round(len(closed_trades) / duration_days, 2) if duration_days > 0 else 0

    return pd.DataFrame(
        [
            {
                "label": label,
                "starting_balance": config.backtest.starting_balance,
                "ending_balance": round(ending_balance, 2),
                "net_pnl": round(ending_balance - config.backtest.starting_balance, 2),
                "closed_trades": len(closed_trades),
                "wins": wins,
                "losses": losses,
                "open_trades": int((trades_df["result"] == "OPEN").sum()) if not trades_df.empty else 0,
                "win_rate_pct": win_rate,
                "true_win_rate_pct": true_win_rate,
                "profit_factor": profit_factor,
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "skipped_overlap_signals": skipped_overlap,
                "trades_per_day": trades_per_day,
            }
        ]
    )


def run_backtest_frame(df, config, label="full_period"):
    df = df.sort_values("time").reset_index(drop=True)

    equity = config.backtest.starting_balance
    peak_equity = equity
    max_drawdown_pct = 0.0
    active_trades = []
    trade_records = []
    skipped_overlap = 0
    reentry_candidates = [] 
    signal_entry_counts = {} # signal_time -> count

    for _, row in df.iterrows():
        current_time = row["time"]

        next_active_trades = []
        for active_trade in active_trades:
            outcome, exit_price, was_be_stop = _resolve_trade(active_trade, row)
            if outcome:
                if outcome == "WIN":
                    r_multiple = (
                        active_trade["reward_distance"]
                        / active_trade["risk_distance"]
                    )
                    pnl = (
                        active_trade["risk_amount"] * r_multiple
                        - config.backtest.commission_per_trade
                    )
                elif outcome == "BE":
                    pnl = -config.backtest.commission_per_trade # Preserve capital
                else:
                    pnl = (
                        -active_trade["risk_amount"]
                        - config.backtest.commission_per_trade
                    )

                equity += pnl
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

                # Only allow re-entry if the market isn't CHOPPY
                if was_be_stop and row["market_state"] != "CHOPPY":
                    reentry_candidates.append(active_trade)

                trade_records.append(
                    {
                        **active_trade,
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "reentry_count": signal_entry_counts.get(active_trade["signal_time"], 1),
                        "result": outcome,
                        "pnl": round(pnl, 2),
                        "equity_after_trade": round(equity, 2),
                    }
                )
            else:
                next_active_trades.append(active_trade)

        active_trades = next_active_trades

        # --- RE-ENTRY LOGIC ---
        # Check if price has reclaimed original entry after a BE stop
        still_watching_reentry = []
        for candidate in reentry_candidates:
            reentry_triggered = False
            if candidate["side"] == "buy" and float(row["close"]) > candidate["executed_entry"]:
                reentry_triggered = True
            elif candidate["side"] == "sell" and float(row["close"]) < candidate["executed_entry"]:
                reentry_triggered = True
            
            # Only allow ONE re-entry per signal and ONLY for ELITE quality.
            # Re-entering MEDIUM trades often leads to the 30% win-rate drawdowns you saw.
            sig_id = candidate["signal_time"]
            if reentry_triggered and signal_entry_counts.get(sig_id, 1) < 2 and candidate["quality"] == "ELITE":
                # Create a fresh copy for the re-entry
                new_trade = candidate.copy()
                new_trade["is_be"] = False # Reset BE status
                new_trade["signal_label"] += "_REENTRY"
                signal_entry_counts[sig_id] = 2
                active_trades.append(new_trade)
            else:
                still_watching_reentry.append(candidate)
        reentry_candidates = still_watching_reentry

        if (
            active_trades
            and not config.backtest.allow_overlapping_positions
        ):
            if row["confirmed_signal"] in {"buy", "sell"}:
                skipped_overlap += 1
            continue

        if row["confirmed_signal"] not in {"buy", "sell"}:
            continue
        signal_entry_counts[row["time"]] = 1

        candidate_trade = _build_trade(row, config, equity)
        if candidate_trade is None:
            continue

        active_trades.append(candidate_trade)

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

    trades_df = pd.DataFrame(trade_records) if trade_records else _empty_trades_frame()
    summary = _build_summary(
        trades_df=trades_df,
        config=config,
        ending_balance=equity,
        max_drawdown_pct=max_drawdown_pct,
        skipped_overlap=skipped_overlap,
        label=label,
    )
    return trades_df, summary


def run(config, in_sample_end: str | None = None, oos_start: str | None = None):
    # 1. Validate dates FIRST
    oos_valid = False
    if in_sample_end and oos_start:
        is_dt, oos_dt = pd.to_datetime(in_sample_end), pd.to_datetime(oos_start)
        if is_dt >= oos_dt:
            logger.warning("⚠️ Invalid OOS dates: IS end must be before OOS start. Skipping OOS mode.")
        else:
            oos_valid = True

    df = pd.read_csv(config.paths.trade_setups, parse_dates=["time"], low_memory=False)

    # 2. Cleanup old split summaries
    for split in ["in_sample_summary.csv", "out_of_sample_summary.csv"]:
        p = config.paths.backtest_summary.parent / split
        if p.exists(): p.unlink()

    # 3. Prepare backtest tasks
    tasks = [("full_period", df, config.paths.backtest_summary)]
    if oos_valid:
        logger.info("Executing Date-Based OOS Split: IS <= %s, OOS >= %s", in_sample_end, oos_start)
        is_df, oos_df = df[df["time"] <= is_dt].copy(), df[df["time"] >= oos_dt].copy()
        if not is_df.empty:
            tasks.append(("in_sample", is_df, config.paths.backtest_summary.parent / "in_sample_summary.csv"))
        if not oos_df.empty:
            tasks.append(("out_of_sample", oos_df, config.paths.backtest_summary.parent / "out_of_sample_summary.csv"))

    # 4. Execute backtests and save
    for label, frame, out_path in tasks:
        trades_df, summary = run_backtest_frame(frame, config, label=label)
        summary.to_csv(out_path, index=False)
        if label == "full_period":
            trades_df.to_csv(config.paths.backtest_trades, index=False)

    logger.info("Backtest trades saved at %s", config.paths.backtest_trades)
    logger.info("Backtest summary saved at %s", config.paths.backtest_summary)
