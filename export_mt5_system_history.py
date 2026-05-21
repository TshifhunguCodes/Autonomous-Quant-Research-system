from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from agents.data_agent import resolve_symbol
from config.v3_config import V3Config
from core.config import load_config


SYSTEM_COMMENT_PREFIX = "AQ_"
DEFAULT_MAGIC = 202404


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export broker-side MT5 trade history opened by the AQRS system."
    )
    parser.add_argument("--config", default=None, help="Optional config JSON path.")
    parser.add_argument("--days", type=int, default=90, help="Lookback window when --from-date is not supplied.")
    parser.add_argument("--from-date", default=None, help="Start date/time, e.g. 2026-05-01 or 2026-05-01T08:00:00.")
    parser.add_argument("--to-date", default=None, help="End date/time. Defaults to now.")
    parser.add_argument("--symbol", default=None, help="Override symbol. Defaults to config market symbol.")
    parser.add_argument("--output", default="data/live/mt5_system_trade_history.csv", help="Output CSV path.")
    parser.add_argument("--comment-prefix", default=SYSTEM_COMMENT_PREFIX, help="System comment prefix to include.")
    parser.add_argument("--magic", type=int, default=None, help="Magic number to include. Defaults to config or AQRS default.")
    parser.add_argument(
        "--include-all-symbols",
        action="store_true",
        help="Do not filter by symbol. Still filters by magic/comment.",
    )
    return parser.parse_args()


def parse_dt(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    return datetime.fromisoformat(value)


def rows_from_mt5_tuples(items) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    return pd.DataFrame([item._asdict() for item in items])


def is_system_entry(row: pd.Series, magic: int, comment_prefix: str) -> bool:
    comment = str(row.get("comment", "") or "")
    row_magic = pd.to_numeric(pd.Series([row.get("magic", None)]), errors="coerce").iloc[0]
    magic_match = pd.notna(row_magic) and int(row_magic) == int(magic)
    comment_match = comment.startswith(comment_prefix)
    return bool(magic_match or comment_match)


def deal_side(deal_type: int) -> str:
    if deal_type == getattr(mt5, "DEAL_TYPE_BUY", 0):
        return "BUY"
    if deal_type == getattr(mt5, "DEAL_TYPE_SELL", 1):
        return "SELL"
    return str(deal_type)


def parse_system_from_comment(comment: str) -> str:
    text = str(comment or "")
    if "ALPHA" in text:
        return "ALPHA"
    if "FLOW" in text:
        return "FLOW_EXP"
    return "UNKNOWN"


def build_position_history(deals_df: pd.DataFrame, magic: int, comment_prefix: str) -> pd.DataFrame:
    if deals_df.empty:
        return pd.DataFrame()

    deals = deals_df.copy()
    deals["time"] = pd.to_datetime(deals["time"], unit="s", errors="coerce")
    for col in ["profit", "commission", "swap", "fee", "volume", "price"]:
        if col in deals.columns:
            deals[col] = pd.to_numeric(deals[col], errors="coerce").fillna(0.0)

    entry_code = getattr(mt5, "DEAL_ENTRY_IN", 0)
    exit_code = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    position_col = "position_id" if "position_id" in deals.columns else "position"

    entries = deals[deals["entry"].eq(entry_code)].copy()
    entries = entries[entries.apply(lambda row: is_system_entry(row, magic, comment_prefix), axis=1)]
    if entries.empty:
        return pd.DataFrame()

    records = []
    for _, entry in entries.sort_values("time").iterrows():
        position_id = entry.get(position_col)
        if pd.isna(position_id):
            continue

        position_deals = deals[deals[position_col].eq(position_id)].copy()
        exits = position_deals[
            position_deals["entry"].eq(exit_code)
            & position_deals["time"].ge(entry["time"])
        ].copy()

        pnl_cols = [col for col in ["profit", "commission", "swap", "fee"] if col in exits.columns]
        realized_pnl = float(exits[pnl_cols].sum().sum()) if pnl_cols and not exits.empty else 0.0
        exit_time = exits["time"].max() if not exits.empty else pd.NaT
        exit_price = float(exits.sort_values("time").iloc[-1]["price"]) if not exits.empty else 0.0
        exit_deals = ",".join(str(int(ticket)) for ticket in exits.get("ticket", pd.Series(dtype=int)).dropna())
        result = "OPEN"
        if not exits.empty:
            result = "WIN" if realized_pnl > 0 else "LOSS" if realized_pnl < 0 else "BREAKEVEN"

        entry_comment = str(entry.get("comment", "") or "")
        records.append(
            {
                "entry_time": entry["time"],
                "exit_time": exit_time,
                "symbol": entry.get("symbol", ""),
                "side": deal_side(int(entry.get("type", -1))),
                "system": parse_system_from_comment(entry_comment),
                "result": result,
                "volume": float(entry.get("volume", 0.0) or 0.0),
                "entry_price": float(entry.get("price", 0.0) or 0.0),
                "exit_price": exit_price,
                "pnl": realized_pnl,
                "position_id": position_id,
                "entry_deal": entry.get("ticket", ""),
                "entry_order": entry.get("order", ""),
                "exit_deals": exit_deals,
                "magic": entry.get("magic", ""),
                "comment": entry_comment,
                "duration_minutes": (
                    round((exit_time - entry["time"]).total_seconds() / 60, 2)
                    if pd.notna(exit_time)
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    base_config = load_config(args.config)
    config = V3Config.load_from(base_config)

    symbol = args.symbol or config.market.symbol
    if not args.include_all_symbols:
        symbol = resolve_symbol(symbol)

    to_dt = parse_dt(args.to_date, datetime.now())
    from_dt = parse_dt(args.from_date, to_dt - timedelta(days=args.days))
    magic = args.magic
    if magic is None:
        magic = int(getattr(config.market, "magic_number", DEFAULT_MAGIC))

    if not mt5.initialize():
        print("ERROR: MT5 initialize failed. Open/login to MT5 terminal and try again.")
        return 1

    try:
        deals = mt5.history_deals_get(from_dt, to_dt)
        deals_df = rows_from_mt5_tuples(deals)
        if deals_df.empty:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(output, index=False)
            print(f"No MT5 deals found in range. Empty CSV written: {output}")
            return 0

        if not args.include_all_symbols and "symbol" in deals_df.columns:
            deals_df = deals_df[deals_df["symbol"].astype(str).eq(symbol)]

        history = build_position_history(deals_df, magic=magic, comment_prefix=args.comment_prefix)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(output, index=False)

        closed = history[history["result"].isin(["WIN", "LOSS", "BREAKEVEN"])] if not history.empty else history
        wins = int((closed["result"] == "WIN").sum()) if not closed.empty else 0
        losses = int((closed["result"] == "LOSS").sum()) if not closed.empty else 0
        net_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0

        print(f"Exported {len(history)} AQRS MT5 positions to {output}")
        print(f"Closed: {len(closed)} | Wins: {wins} | Losses: {losses} | Net PnL: {net_pnl:.2f}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
