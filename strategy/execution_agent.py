import math

import pandas as pd

from core.logging_utils import get_logger

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - depends on local MT5 install
    mt5 = None


logger = get_logger(__name__)


def _load_execution_log(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=[
            "signal_time",
            "symbol",
            "side",
            "quality",
            "confirm_score",
            "market_state",
            "deal",
            "price",
        ]
    )


def _append_execution_log(config, signal_row, deal, price):
    log_df = _load_execution_log(config.paths.execution_log)
    log_df = pd.concat(
        [
            log_df,
            pd.DataFrame(
                [
                    {
                        "signal_time": signal_row["time"],
                        "symbol": config.market.symbol,
                        "side": signal_row["confirmed_signal"],
                        "quality": signal_row["quality"],
                        "confirm_score": signal_row["confirm_score"],
                        "market_state": signal_row["market_state"],
                        "deal": deal,
                        "price": price,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    log_df.to_csv(config.paths.execution_log, index=False)


def _validate_live_signal(config, signal_row):
    reasons = []

    signal_type = signal_row["confirmed_signal"]
    if signal_type not in {"buy", "sell"}:
        reasons.append("latest row is not a confirmed trade signal")

    if signal_row.get("quality") not in set(config.live.approved_qualities):
        reasons.append("quality is below the live-approved threshold")

    if float(signal_row.get("confirm_score", 0)) < config.live.min_confirm_score:
        reasons.append("confirm score is below the configured minimum")

    if config.live.require_h1_alignment and int(signal_row.get("h1_alignment", 0)) != 1:
        reasons.append("H1 bias is not aligned with the M5 entry")

    if signal_row.get("market_state") in set(config.live.disallowed_market_states):
        reasons.append("market state is blocked for live trading")

    spread_value = float(signal_row.get("spread", 0) or 0)
    if spread_value > config.live.max_spread_points:
        reasons.append("spread is above the configured live maximum")

    for column in ["entry_price", "stop_loss", "take_profit"]:
        value = signal_row.get(column)
        if pd.isna(value) or not math.isfinite(float(value)):
            reasons.append(f"{column} is missing or invalid")

    signal_time = pd.to_datetime(signal_row["time"])
    age_minutes = (pd.Timestamp.utcnow().tz_localize(None) - signal_time).total_seconds() / 60
    if age_minutes > config.live.max_signal_age_minutes:
        reasons.append("signal is stale for live execution")

    execution_log = _load_execution_log(config.paths.execution_log)
    if (
        not config.live.allow_duplicate_candle
        and not execution_log.empty
        and str(signal_row["time"]) in execution_log["signal_time"].astype(str).values
    ):
        reasons.append("a trade was already logged for this candle")

    return reasons


def _preview_payload(config, signal_row, reasons):
    payload = {
        "symbol": config.market.symbol,
        "time": str(signal_row["time"]),
        "side": signal_row["confirmed_signal"],
        "quality": signal_row["quality"],
        "confirm_score": float(signal_row["confirm_score"]),
        "market_state": signal_row["market_state"],
        "entry_price": float(signal_row["entry_price"]),
        "stop_loss": float(signal_row["stop_loss"]),
        "take_profit": float(signal_row["take_profit"]),
        "blocked_reasons": reasons,
    }
    logger.info("Live preview | %s", payload)
    return payload


def run(config, execute=False):
    df = pd.read_csv(config.paths.trade_setups, parse_dates=["time"])
    if df.empty:
        logger.warning("No trade setups found for live mode")
        return {"status": "empty"}

    last_signal = df.iloc[-1].copy()
    reasons = _validate_live_signal(config, last_signal)
    preview = _preview_payload(config, last_signal, reasons)

    if not execute:
        logger.warning("Live mode is running in preview-only mode")
        return {"status": "preview", "preview": preview}

    if not config.live.enabled:
        logger.error("Live execution is disabled in config/app_config.json")
        return {"status": "blocked", "preview": preview}

    if reasons:
        logger.warning("Live execution blocked: %s", "; ".join(reasons))
        return {"status": "blocked", "preview": preview}

    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is not installed. Install dependencies before live execution."
        )

    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed in execution agent")

    try:
        symbol = config.market.symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"MT5 symbol_info returned None for {symbol}")

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions:
            logger.warning("Live execution blocked: open position already exists for %s", symbol)
            return {"status": "blocked", "preview": preview}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to read tick data for {symbol}")

        signal_type = last_signal["confirmed_signal"]
        price = tick.ask if signal_type == "buy" else tick.bid
        type_order = (
            mt5.ORDER_TYPE_BUY if signal_type == "buy" else mt5.ORDER_TYPE_SELL
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": config.live.lot,
            "type": type_order,
            "price": price,
            "sl": float(last_signal["stop_loss"]),
            "tp": float(last_signal["take_profit"]),
            "magic": 123456,
            "comment": f"AutoQuantV2_{last_signal['quality']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(
            "Sending live order | side=%s symbol=%s quality=%s",
            signal_type,
            symbol,
            last_signal["quality"],
        )

        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError("MT5 order_send returned None")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Live order failed with retcode=%s", result.retcode)
            return {"status": "error", "retcode": result.retcode, "preview": preview}

        _append_execution_log(config, last_signal, result.deal, price)
        logger.info("Live trade opened successfully: deal=%s", result.deal)
        return {"status": "executed", "deal": result.deal, "preview": preview}
    finally:
        mt5.shutdown()
