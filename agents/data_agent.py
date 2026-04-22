import pandas as pd
from datetime import datetime, timedelta, timezone

from core.logging_utils import get_logger

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - depends on local MT5 install
    mt5 = None


logger = get_logger(__name__)


def get_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"MT5 returned no data for {symbol} timeframe={timeframe}")

    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"MT5 returned an empty dataset for {symbol}")

    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def resolve_symbol(preferred_symbol):
    candidates = [preferred_symbol]
    all_symbols = mt5.symbols_get() or []
    xau_aliases = [
        symbol.name
        for symbol in all_symbols
        if preferred_symbol.upper() in symbol.name.upper()
    ]
    for alias in xau_aliases:
        if alias not in candidates:
            candidates.append(alias)

    for candidate in candidates:
        if mt5.symbol_select(candidate, True):
            return candidate

    raise RuntimeError(f"Unable to select a tradable MT5 symbol for {preferred_symbol}")


def get_data_range(symbol, timeframe, start_utc, end_utc):
    rates = mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
    if rates is None:
        raise RuntimeError(f"MT5 returned no data for {symbol} timeframe={timeframe}")

    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"MT5 returned an empty dataset for {symbol}")

    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def get_data_range_chunked(symbol, timeframe, start_utc, end_utc, chunk_days):
    chunks = []
    current_start = start_utc
    while current_start < end_utc:
        current_end = min(current_start + timedelta(days=chunk_days), end_utc)
        chunk = get_data_range(symbol, timeframe, current_start, current_end)
        chunks.append(chunk)
        current_start = current_end

    return (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def merge_existing_data(path, df_new):
    frames = [df_new]
    if path.exists():
        frames.append(pd.read_csv(path, parse_dates=["time"]))

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    merged.to_csv(path, index=False)
    return merged


def run(config):
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is not installed. Install dependencies before refreshing MT5 data."
        )

    symbol = config.market.symbol

    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed in data agent")

    try:
        resolved_symbol = resolve_symbol(symbol)
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=int(config.market.history_years * 365))

        df_m5 = get_data_range_chunked(
            resolved_symbol,
            mt5.TIMEFRAME_M5,
            start_utc,
            end_utc,
            chunk_days=120,
        )
        df_h1 = get_data_range(resolved_symbol, mt5.TIMEFRAME_H1, start_utc, end_utc)

        df_m5 = merge_existing_data(config.paths.raw_m5, df_m5)
        df_h1 = merge_existing_data(config.paths.raw_h1, df_h1)
    finally:
        mt5.shutdown()

    logger.info(
        "Raw MT5 data saved for %s: M5 rows=%s, H1 rows=%s",
        resolved_symbol,
        len(df_m5),
        len(df_h1),
    )
