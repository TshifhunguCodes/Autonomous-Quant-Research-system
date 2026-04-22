import pandas as pd

from core.logging_utils import get_logger


logger = get_logger(__name__)


def _build_trade(row, config, equity):
    if not bool(row.get("trade_allowed", True)):
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
        * float(row.get("risk_multiplier", 1.0) or 1.0),
        "confirm_score": row.get("confirm_score", 0),
    }


def _resolve_trade(trade, candle):
    high = float(candle["high"])
    low = float(candle["low"])

    if trade["side"] == "buy":
        if low <= trade["stop_loss"] and high >= trade["take_profit"]:
            return "LOSS", trade["stop_loss"]
        if low <= trade["stop_loss"]:
            return "LOSS", trade["stop_loss"]
        if high >= trade["take_profit"]:
            return "WIN", trade["take_profit"]
    else:
        if high >= trade["stop_loss"] and low <= trade["take_profit"]:
            return "LOSS", trade["stop_loss"]
        if high >= trade["stop_loss"]:
            return "LOSS", trade["stop_loss"]
        if low <= trade["take_profit"]:
            return "WIN", trade["take_profit"]

    return None, None


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
            "pnl",
            "equity_after_trade",
        ]
    )


def _build_summary(trades_df, config, ending_balance, max_drawdown_pct, skipped_overlap, label):
    closed_trades = (
        trades_df[trades_df["result"].isin(["WIN", "LOSS"])]
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
                "profit_factor": profit_factor,
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "skipped_overlap_signals": skipped_overlap,
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

    for _, row in df.iterrows():
        current_time = row["time"]

        next_active_trades = []
        for active_trade in active_trades:
            outcome, exit_price = _resolve_trade(active_trade, row)
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

                trade_records.append(
                    {
                        **active_trade,
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "result": outcome,
                        "pnl": round(pnl, 2),
                        "equity_after_trade": round(equity, 2),
                    }
                )
            else:
                next_active_trades.append(active_trade)

        active_trades = next_active_trades

        if (
            active_trades
            and not config.backtest.allow_overlapping_positions
        ):
            if row["confirmed_signal"] in {"buy", "sell"}:
                skipped_overlap += 1
            continue

        if row["confirmed_signal"] not in {"buy", "sell"}:
            continue

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


def run(config):
    df = pd.read_csv(config.paths.trade_setups, parse_dates=["time"], low_memory=False)
    trades_df, summary = run_backtest_frame(df, config, label="full_period")

    trades_df.to_csv(config.paths.backtest_trades, index=False)
    summary.to_csv(config.paths.backtest_summary, index=False)

    logger.info("Backtest trades saved at %s", config.paths.backtest_trades)
    logger.info("Backtest summary saved at %s", config.paths.backtest_summary)
