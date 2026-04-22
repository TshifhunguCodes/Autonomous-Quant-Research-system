import pandas as pd

from core.logging_utils import get_logger


logger = get_logger(__name__)


def _read_csv_if_exists(path, **kwargs):
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return pd.DataFrame()


def _format_terminal_summary(stats, summary_rows: list[pd.Series], session_df, top_setup, weak_setup):
    lines = [
        "",
        "IMPORTANT SUMMARY",
        f"Setups generated: {stats['setups']}",
        f"Confirmed trades: {stats['confirmed']}",
        f"Entries planned: {stats['entries']}",
        f"Quality mix: ELITE={stats['elite_count']} HIGH={stats['high_count']} MEDIUM={stats['medium_count']}",
    ]

    for summary_row in summary_rows:
        label = summary_row.get('label', 'UNKNOWN').replace('_', ' ').upper()
        lines.append(f"\n--- {label} ---")
        lines.extend(
            [
                f"Wins / Losses: {int(summary_row['wins'])} / {int(summary_row['losses'])}",
                f"Standard Win rate: {summary_row['win_rate_pct']}% (includes BE)",
                f"True Win rate: {summary_row['true_win_rate_pct']}% (Accuracy)",
                f"Net PnL: {summary_row['net_pnl']}",
                f"Profit factor: {summary_row['profit_factor']}",
                f"Max drawdown: {summary_row['max_drawdown_pct']}%",
                f"Frequency: {summary_row.get('trades_per_day', 0)} trades/day",
            ]
        )

    lines.append("")
    lines.append("ANALYSIS")

    if session_df is not None and not session_df.empty:
        for _, sess in session_df.iterrows():
            lines.append(
                f"Session: {sess['session']:<12} | trades={int(sess['closed_trades']):<4} | net_pnl={sess['net_pnl']:<10} | win_rate={sess['win_rate_pct']}%"
            )

    if top_setup is not None:
        lines.append(
            f"Best setup: {top_setup['setup']} {top_setup['side']} {top_setup['quality']} {top_setup['market_state']} | trades={int(top_setup['closed_trades'])} | net_pnl={top_setup['net_pnl']}"
        )

    if weak_setup is not None:
        lines.append(
            f"Weakest setup: {weak_setup['setup']} {weak_setup['side']} {weak_setup['quality']} {weak_setup['market_state']} | trades={int(weak_setup['closed_trades'])} | net_pnl={weak_setup['net_pnl']}"
        )

    return "\n".join(lines)


def run(config, print_terminal: bool = False):
    df = pd.read_csv(config.paths.trade_setups, parse_dates=["time"], low_memory=False)

    stats = {
        "zone_hits": int(
            ((df["near_support"] == 1) | (df["near_resistance"] == 1)).sum()
        ),
        "setups": int(df["setup"].isin(["BUY_SETUP", "SELL_SETUP"]).sum()),
        "confirmed": int(df["confirmed_signal"].isin(["buy", "sell"]).sum()),
        "entries": int(df["entry_price"].notna().sum()),
        "h1_aligned": int(df.get("h1_alignment", pd.Series(dtype=int)).sum())
        if "h1_alignment" in df.columns
        else 0,
        "elite_count": int((df["quality"] == "ELITE").sum()),
        "high_count": int((df["quality"] == "HIGH").sum()),
        "medium_count": int((df["quality"] == "MEDIUM").sum()),
    }

    logger.info(
        "Pipeline report | zone_hits=%s setups=%s confirmed=%s entries=%s h1_aligned=%s elite=%s high=%s medium=%s",
        stats["zone_hits"],
        stats["setups"],
        stats["confirmed"],
        stats["entries"],
        stats["h1_aligned"],
        stats["elite_count"],
        stats["high_count"],
        stats["medium_count"],
    )

    summary_rows = []
    for fname in ["backtest_summary.csv", "in_sample_summary.csv", "out_of_sample_summary.csv"]:
        path = config.paths.backtest_summary.parent / fname
        if path.exists():
            summary = pd.read_csv(path)
            if not summary.empty:
                summary_rows.append(summary.iloc[0])

    session_df = _read_csv_if_exists(config.paths.session_performance)

    setup_df = _read_csv_if_exists(config.paths.best_setup_types)
    top_setup = setup_df.iloc[0] if not setup_df.empty else None
    weak_setup = setup_df.sort_values("net_pnl").iloc[0] if not setup_df.empty else None

    if print_terminal:
        print(
            _format_terminal_summary(
                stats=stats,
                summary_rows=summary_rows,
                session_df=session_df,
                top_setup=top_setup,
                weak_setup=weak_setup,
            )
        )

    return stats
