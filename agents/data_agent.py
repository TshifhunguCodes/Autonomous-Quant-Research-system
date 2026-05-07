import pandas as pd
from datetime import datetime, timedelta, timezone

import time # Added import
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
    """Resolve and select a tradable MT5 symbol with fallback to common variants."""
    candidates = [preferred_symbol]
    
    # Add common XAUUSD variants that brokers use
    common_variants = [
        "XAUUSDm",    # With commission
        "XAUUSD.a",   # ECN account
        "XAUUSD.b",   # Another variant
        "XAUUSD.c",   # Another variant
        "GOLD",       # Some brokers use this
        "Gold",       # Capitalized variant
        "gold",       # Lowercase variant
        "XAU_USD",    # With underscore
        "XAU-USD",    # With dash
    ]
    
    # Ensure MT5 is initialized before getting symbols
    if not mt5.initialize():
        logger.error("MT5 initialization failed in resolve_symbol")
        raise RuntimeError("Failed to initialize MT5 for symbol resolution")
    
    # Get all available symbols from MT5
    all_symbols = mt5.symbols_get()
    if all_symbols is None or len(all_symbols) == 0:
        logger.warning("mt5.symbols_get() returned empty list. Trying direct symbol info lookup...")
        all_symbols = []
    
    # Build comprehensive candidate list
    # 1. First add exact matches and partial matches from MT5
    xau_aliases = [
        symbol.name
        for symbol in all_symbols
        if preferred_symbol.upper() in symbol.name.upper() or 
           symbol.name.upper() in preferred_symbol.upper()
    ]
    for alias in xau_aliases:
        if alias not in candidates:
            candidates.append(alias)
    
    # 2. Then add common variants that exist in MT5
    for variant in common_variants:
        if variant not in candidates:
            # Check if this variant exists in MT5
            symbol_info = mt5.symbol_info(variant)
            if symbol_info is not None:
                candidates.append(variant)
    
    # 3. If still only have the original, try to get symbol info for the preferred symbol
    if len(candidates) == 1:
        symbol_info = mt5.symbol_info(preferred_symbol)
        if symbol_info is not None:
            logger.info(f"Found symbol info for {preferred_symbol}: visible={symbol_info.visible}, "
                       f"tradable={symbol_info.trade_mode}")
            if symbol_info.visible:
                # Symbol exists and is visible, try to select it directly
                if mt5.symbol_select(preferred_symbol, True):
                    test_data = mt5.copy_rates_from_pos(preferred_symbol, mt5.TIMEFRAME_M5, 0, 1)
                    if test_data is not None and len(test_data) > 0:
                        logger.info(f"✅ Successfully selected MT5 symbol: {preferred_symbol}")
                        return preferred_symbol
    
    logger.info(f"Attempting to resolve symbol '{preferred_symbol}'. Candidates: {candidates}")
    
    for attempt in range(1, 4):  # Retry up to 3 times
        for candidate in candidates:
            try:
                # First ensure the symbol is selected in Market Watch
                if mt5.symbol_select(candidate, True):
                    # Verify we can actually get data for this symbol
                    test_data = mt5.copy_rates_from_pos(candidate, mt5.TIMEFRAME_M5, 0, 1)
                    if test_data is not None and len(test_data) > 0:
                        logger.info(f"✅ Successfully selected and verified MT5 symbol: {candidate}")
                        return candidate
                    else:
                        logger.debug(f"Symbol {candidate} selected but returned no data")
                else:
                    logger.debug(f"Failed to select candidate: {candidate}")
            except Exception as e:
                logger.debug(f"Error testing candidate {candidate}: {e}")
                continue
        
        if attempt < 3:
            logger.warning(f"Attempt {attempt}/3: Failed to select any tradable MT5 symbol for '{preferred_symbol}'. "
                          f"Retrying in 2 seconds...")
            time.sleep(2)
        else:
            # Final attempt - provide detailed diagnostic information
            logger.error(f"❌ Failed to select any symbol after 3 attempts.")
            logger.error(f"Total candidates tried: {len(candidates)}")
            logger.error(f"Available symbols count: {len(all_symbols)}")
            if all_symbols:
                logger.error(f"First 20 available symbols: {[s.name for s in all_symbols[:20]]}")
            logger.error(f"Please ensure XAUUSD (or a variant) is visible in MT5 Market Watch.")
            logger.error(f"Steps to fix:")
            logger.error(f"1. Open MT5 terminal")
            logger.error(f"2. Press Ctrl+U to open Symbols window")
            logger.error(f"3. Find and enable XAUUSD, XAUUSDm, GOLD, or similar")
            logger.error(f"4. Ensure it appears in Market Watch (Ctrl+M)")
    
    raise RuntimeError(f"Persistent failure: Unable to select a tradable MT5 symbol for '{preferred_symbol}' after multiple attempts. "
                       "Verify MT5 terminal is running, logged in, and the symbol is in Market Watch.")



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
