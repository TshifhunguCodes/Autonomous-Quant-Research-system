import pandas as pd

from core.logging_utils import get_logger
from strategy.backtesting import run_backtest_frame


logger = get_logger(__name__)


def _build_rolling_windows(df, window_days, step_days):
    if df.empty:
        return []

    start = df["time"].min().normalize()
    last = df["time"].max()
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)

    windows = []
    current_start = start
    while current_start <= last:
        current_end = current_start + window
        window_df = df[(df["time"] >= current_start) & (df["time"] < current_end)].copy()
        if not window_df.empty:
            windows.append((current_start, current_end, window_df))
        current_start += step
    return windows


def _summarize_monthly(trades_df):
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "closed_trades",
                "wins",
                "losses",
                "win_rate_pct",
                "net_pnl",
                "gross_profit",
                "gross_loss",
                "profit_factor",
            ]
        )

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "closed_trades",
                "wins",
                "losses",
                "win_rate_pct",
                "net_pnl",
                "gross_profit",
                "gross_loss",
                "profit_factor",
            ]
        )

    closed["month"] = pd.to_datetime(closed["exit_time"]).dt.to_period("M").astype(str)

    rows = []
    for month, group in closed.groupby("month"):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        gross_profit = float(group.loc[group["pnl"] > 0, "pnl"].sum())
        gross_loss = float(group.loc[group["pnl"] < 0, "pnl"].sum())
        profit_factor = round(gross_profit / abs(gross_loss), 2) if gross_loss else 0.0
        win_rate = round((wins / len(group)) * 100, 2) if len(group) else 0.0
        rows.append(
            {
                "month": month,
                "closed_trades": len(group),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": win_rate,
                "net_pnl": round(float(group["pnl"].sum()), 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "profit_factor": profit_factor,
            }
        )

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _build_equity_curve(trades_df, starting_balance):
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "trade_number",
                "signal_time",
                "exit_time",
                "side",
                "setup",
                "quality",
                "result",
                "pnl",
                "equity",
                "running_max_equity",
                "drawdown_pct",
            ]
        )

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return pd.DataFrame(
            columns=[
                "trade_number",
                "signal_time",
                "exit_time",
                "side",
                "setup",
                "quality",
                "result",
                "pnl",
                "equity",
                "running_max_equity",
                "drawdown_pct",
            ]
        )

    closed = closed.sort_values(["exit_time", "signal_time"]).reset_index(drop=True)
    closed["trade_number"] = range(1, len(closed) + 1)
    closed["equity"] = starting_balance + closed["pnl"].cumsum()
    closed["running_max_equity"] = closed["equity"].cummax()
    closed["drawdown_pct"] = (
        (closed["running_max_equity"] - closed["equity"])
        / closed["running_max_equity"]
        * 100
    ).round(2)

    return closed[
        [
            "trade_number",
            "signal_time",
            "exit_time",
            "side",
            "setup",
            "quality",
            "result",
            "pnl",
            "equity",
            "running_max_equity",
            "drawdown_pct",
        ]
    ]


def _build_losing_streak_reports(trades_df):
    streak_columns = [
        "streak_id",
        "start_signal_time",
        "end_signal_time",
        "start_exit_time",
        "end_exit_time",
        "loss_count",
        "total_pnl",
    ]
    summary_columns = [
        "streak_count",
        "max_losing_streak",
        "average_losing_streak",
        "current_losing_streak",
        "worst_streak_pnl",
    ]

    if trades_df.empty:
        return pd.DataFrame(columns=streak_columns), pd.DataFrame(columns=summary_columns)

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=streak_columns), pd.DataFrame(columns=summary_columns)

    closed = closed.sort_values(["exit_time", "signal_time"]).reset_index(drop=True)

    streak_rows = []
    current_losses = []
    streak_id = 0

    def flush_streak(loss_rows):
        nonlocal streak_id
        if not loss_rows:
            return
        streak_id += 1
        streak_df = pd.DataFrame(loss_rows)
        streak_rows.append(
            {
                "streak_id": streak_id,
                "start_signal_time": streak_df["signal_time"].iloc[0],
                "end_signal_time": streak_df["signal_time"].iloc[-1],
                "start_exit_time": streak_df["exit_time"].iloc[0],
                "end_exit_time": streak_df["exit_time"].iloc[-1],
                "loss_count": len(streak_df),
                "total_pnl": round(float(streak_df["pnl"].sum()), 2),
            }
        )

    for _, row in closed.iterrows():
        if row["result"] == "LOSS":
            current_losses.append(row.to_dict())
        else:
            flush_streak(current_losses)
            current_losses = []

    flush_streak(current_losses)

    streaks_df = pd.DataFrame(streak_rows, columns=streak_columns)
    current_losing_streak = 0
    for result in reversed(closed["result"].tolist()):
        if result == "LOSS":
            current_losing_streak += 1
        else:
            break

    if streaks_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "streak_count": 0,
                    "max_losing_streak": 0,
                    "average_losing_streak": 0.0,
                    "current_losing_streak": current_losing_streak,
                    "worst_streak_pnl": 0.0,
                }
            ]
        )
        return streaks_df, summary

    summary = pd.DataFrame(
        [
            {
                "streak_count": len(streaks_df),
                "max_losing_streak": int(streaks_df["loss_count"].max()),
                "average_losing_streak": round(float(streaks_df["loss_count"].mean()), 2),
                "current_losing_streak": current_losing_streak,
                "worst_streak_pnl": round(float(streaks_df["total_pnl"].min()), 2),
            }
        ]
    )
    return streaks_df, summary


def _classify_session(timestamp):
    hour = pd.Timestamp(timestamp).hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 18:
        return "NEW_YORK"
    return "LATE_SESSION"


def _summarize_session_performance(trades_df):
    columns = [
        "session",
        "closed_trades",
        "wins",
        "losses",
        "win_rate_pct",
        "net_pnl",
        "avg_pnl",
        "profit_factor",
    ]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=columns)

    closed["session"] = closed["signal_time"].apply(_classify_session)
    rows = []
    for session, group in closed.groupby("session"):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        gross_profit = float(group.loc[group["pnl"] > 0, "pnl"].sum())
        gross_loss = float(group.loc[group["pnl"] < 0, "pnl"].sum())
        profit_factor = round(gross_profit / abs(gross_loss), 2) if gross_loss else 0.0
        rows.append(
            {
                "session": session,
                "closed_trades": len(group),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round((wins / len(group)) * 100, 2) if len(group) else 0.0,
                "net_pnl": round(float(group["pnl"].sum()), 2),
                "avg_pnl": round(float(group["pnl"].mean()), 2),
                "profit_factor": profit_factor,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["net_pnl", "win_rate_pct"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _summarize_best_setup_types(trades_df):
    columns = [
        "setup",
        "side",
        "quality",
        "market_state",
        "closed_trades",
        "wins",
        "losses",
        "win_rate_pct",
        "net_pnl",
        "avg_pnl",
        "profit_factor",
    ]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    group_cols = ["setup", "side", "quality", "market_state"]
    for keys, group in closed.groupby(group_cols):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        gross_profit = float(group.loc[group["pnl"] > 0, "pnl"].sum())
        gross_loss = float(group.loc[group["pnl"] < 0, "pnl"].sum())
        profit_factor = round(gross_profit / abs(gross_loss), 2) if gross_loss else 0.0
        rows.append(
            {
                "setup": keys[0],
                "side": keys[1],
                "quality": keys[2],
                "market_state": keys[3],
                "closed_trades": len(group),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round((wins / len(group)) * 100, 2) if len(group) else 0.0,
                "net_pnl": round(float(group["pnl"].sum()), 2),
                "avg_pnl": round(float(group["pnl"].mean()), 2),
                "profit_factor": profit_factor,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["net_pnl", "profit_factor", "win_rate_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_system_regime_performance(trades_df):
    """Computes granular performance metrics for A vs B systems across regimes."""
    if trades_df.empty:
        return pd.DataFrame()

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS", "BE"])].copy()
    if closed.empty:
        return pd.DataFrame()
 
    results = []
    # Group by System and various Regimes
    regime_cols = ["session", "market_state", "market_regime"]
    
    for system in closed["system"].unique():
        sys_df = closed[closed["system"] == system]
        
        for col in regime_cols:
            for val, group in sys_df.groupby(col):
                wins = int((group["result"] == "WIN").sum())
                losses = int((group["result"] == "LOSS").sum())
                net_pnl = float(group["pnl"].sum())
                gp = float(group.loc[group["pnl"] > 0, "pnl"].sum())
                gl = abs(float(group.loc[group["pnl"] < 0, "pnl"].sum()))
                
                # Sub-Equity Curve for Drawdown calculation
                curve = group.sort_values("exit_time")["pnl"].cumsum()
                max_dd = 0
                if not curve.empty:
                    peak = curve.cummax()
                    dd = peak - curve
                    max_dd = dd.max()

                results.append({
                    "system": system,
                    "regime_type": col,
                    "regime_value": val,
                    "trades": len(group),
                    "win_rate": round((wins / (wins + losses)) * 100, 2) if (wins + losses) > 0 else 0,
                    "profit_factor": round(gp / gl, 2) if gl > 0 else (round(gp, 2) if gp > 0 else 1.0),
                    "net_pnl": round(net_pnl, 2),
                    "max_dd_points": round(max_dd, 2)
                })
                
    return pd.DataFrame(results).sort_values(["system", "regime_type", "net_pnl"], ascending=[True, True, False])


def _generate_ny_deep_dive(trades_df):
    """Performs granular analysis of the New York session to identify micro-strategies."""
    ny = trades_df[trades_df["session"] == "NEW_YORK"].copy()
    closed_ny = ny[ny["result"].isin(["WIN", "LOSS", "BE"])].copy()
    if closed_ny.empty:
        return "No closed New York trades to analyze."

    # Calculate Duration
    closed_ny["duration_mins"] = (pd.to_datetime(closed_ny["exit_time"]) - pd.to_datetime(closed_ny["signal_time"])).dt.total_seconds() / 60
    avg_dur = closed_ny["duration_mins"].mean()

    # Helper for stats
    def get_stats(df):
        wins = (df["result"] == "WIN").sum()
        losses = (df["result"] == "LOSS").sum()
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0
        return f"Trades: {len(df):<3} | WR: {wr:>5.1f}% | PnL: {df['pnl'].sum():>8.2f}"

    closed_ny["hour"] = pd.to_datetime(closed_ny["signal_time"]).dt.hour
    
    lines = ["", "="*60, "NEW YORK SESSION DEEP DIVE", "="*60]
    lines.append(f"Average Trade Duration: {avg_dur:.1f} minutes")
    
    lines.append("\n--- Performance by Hour (NY) ---")
    for hr, group in closed_ny.groupby("hour"):
        lines.append(f"Hour {hr:02}:00 | {get_stats(group)}")

    lines.append("\n--- Performance by Setup (NY) ---")
    for setup, group in closed_ny.groupby("setup"):
        lines.append(f"Setup {setup:<10} | {get_stats(group)}")

    lines.append("\n--- Performance by Volatility Regime (NY) ---")
    for state, group in closed_ny.groupby("market_state"):
        lines.append(f"Regime {state:<10} | {get_stats(group)}")

    lines.append("\n--- First Breakout vs Later Entries (NY) ---")
    for is_first, group in closed_ny.groupby("is_first_breakout"):
        label = "First Breakout" if is_first else "Later Entry  "
        lines.append(f"{label} | {get_stats(group)}")
    
    lines.append("="*60)
    return "\n".join(lines)


def run(config, rolling_window_days=30, rolling_step_days=7):
    df = pd.read_csv(config.paths.trade_setups, parse_dates=["time"], low_memory=False)
    windows = _build_rolling_windows(
        df,
        window_days=rolling_window_days,
        step_days=rolling_step_days,
    )

    stability_stats = []
    for system_label in ["ALPHA", "FLOW", "COMBINED"]:
        rolling_rows = []
        for start, end, window_df in windows:
            _, summary = run_backtest_frame(
                window_df,
                config,
                label=f"{system_label}_{start.date()}",
                mode=system_label
            )
            row = summary.iloc[0].to_dict()
            row["window_start"] = start
            row["window_end"] = end
            rolling_rows.append(row)

        rolling_df = pd.DataFrame(rolling_rows)
        out_name = f"rolling_{system_label.lower()}_summary.csv"
        rolling_df.to_csv(config.paths.backtest_dir / out_name, index=False)

        # Calculate Stability Metrics
        if not rolling_df.empty and len(rolling_df) > 1:
            wr_std = rolling_df["true_win_rate_pct"].std()
            pf_mean = rolling_df["profit_factor"].mean()
            pf_std = rolling_df["profit_factor"].std()
            dd_mean = rolling_df["max_drawdown_pct"].mean()
            dd_std = rolling_df["max_drawdown_pct"].std()
            
            stability_stats.append({
                "System": system_label,
                "WR_StdDev": round(wr_std, 2),
                "PF_Stability": round(1 - (pf_std / pf_mean), 2) if pf_mean > 0 else 0,
                "DD_Consistency": round(1 - (dd_std / dd_mean), 2) if dd_mean > 0 else 0,
                "Avg_Net_PnL": round(rolling_df["net_pnl"].mean(), 2)
            })

    if stability_stats:
        stability_df = pd.DataFrame(stability_stats)
        stability_path = config.paths.backtest_dir / "consolidated_stability_report.csv"
        stability_df.to_csv(stability_path, index=False)
        
        print("\n" + "="*95)
        print(f"{'CONSOLIDATED STABILITY REPORT (ROLLING WINDOWS)':^95}")
        print("="*95)
        print(stability_df.to_string(index=False))
        print("="*95)

    full_trades = pd.read_csv(config.paths.backtest_trades, parse_dates=["signal_time", "exit_time"])
    
    regime_perf = _summarize_system_regime_performance(full_trades)
    regime_perf.to_csv(config.paths.backtest_dir / "system_regime_performance.csv", index=False)

    # Generate NY Before-vs-After Metrics for terminal display
    ny_perf = regime_perf[regime_perf["regime_value"] == "NEW_YORK"].copy()
    if not ny_perf.empty:
        print("\n" + "-"*40)
        print(" NEW YORK PERFORMANCE (AFTER SESSION REFACTOR) ")
        print("-"*40)
        print(ny_perf[["system", "trades", "win_rate", "profit_factor", "net_pnl"]].to_string(index=False))
        print("-"*40)

    ny_deep_dive = _generate_ny_deep_dive(full_trades)
    print(ny_deep_dive)

    monthly_df = _summarize_monthly(full_trades)
    monthly_df.to_csv(config.paths.monthly_performance, index=False)
    equity_curve_df = _build_equity_curve(full_trades, config.backtest.starting_balance)
    equity_curve_df.to_csv(config.paths.equity_curve, index=False)
    losing_streaks_df, losing_streak_stats_df = _build_losing_streak_reports(full_trades)
    losing_streaks_df.to_csv(config.paths.losing_streaks, index=False)
    losing_streak_stats_df.to_csv(config.paths.losing_streak_stats, index=False)
    session_df = _summarize_session_performance(full_trades)
    session_df.to_csv(config.paths.session_performance, index=False)
    setup_types_df = _summarize_best_setup_types(full_trades)
    setup_types_df.to_csv(config.paths.best_setup_types, index=False)

    logger.info(
        "Rolling backtest summary saved at %s",
        config.paths.rolling_backtest_summary,
    )
    logger.info(
        "Monthly performance report saved at %s",
        config.paths.monthly_performance,
    )
    logger.info("Equity curve saved at %s", config.paths.equity_curve)
    logger.info("Losing streak reports saved at %s and %s", config.paths.losing_streaks, config.paths.losing_streak_stats)
    logger.info("Session performance saved at %s", config.paths.session_performance)
    logger.info("Best setup types saved at %s", config.paths.best_setup_types)
