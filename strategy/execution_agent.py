import os
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import requests

from core.logging_utils import get_logger
from strategy.execution_gate import ExecutionGate
from strategy.mt5_bridge import MT5Bridge

logger = get_logger(__name__)


def _telegram_credentials(config):
    telegram_config = getattr(config, "telegram", None)
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_TOKEN")
        or getattr(telegram_config, "bot_token", "")
        or getattr(config, "TELEGRAM_BOT_TOKEN", "")
        or getattr(config, "TELEGRAM_TOKEN", "")
    )
    chat_id = (
        os.getenv("TELEGRAM_CHAT_ID")
        or getattr(telegram_config, "chat_id", "")
        or getattr(config, "TELEGRAM_CHAT_ID", "")
    )
    return token, chat_id


def send_telegram_msg(config, text):
    """Send a Telegram message when bot token and chat id are configured."""
    token, chat_id = _telegram_credentials(config)

    if not token or not chat_id:
        return

    logger.info("Attempting Telegram message to %s...", chat_id)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
        response.raise_for_status()
    except Exception as e:
        logger.error("Telegram Error: %s", e)


def test_telegram_connection(config):
    """Verify Telegram bot credentials and send a test message."""
    token, chat_id = _telegram_credentials(config)
    if not token or not chat_id:
        return {
            "ok": False,
            "reason": "Missing TELEGRAM_BOT_TOKEN/TELEGRAM_TOKEN or TELEGRAM_CHAT_ID.",
        }

    try:
        bot_response = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=10,
        )
        bot_payload = bot_response.json()
        if not bot_payload.get("ok"):
            return {
                "ok": False,
                "reason": f"Bot token rejected: {bot_payload}",
            }

        message_response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": "AQRS V3 Telegram test: bot is connected and ready.",
            },
            timeout=10,
        )
        message_payload = message_response.json()
        if not message_payload.get("ok"):
            return {
                "ok": False,
                "reason": f"Chat/message rejected: {message_payload}",
                "bot": bot_payload.get("result", {}),
            }

        return {
            "ok": True,
            "bot": bot_payload.get("result", {}),
            "chat_id": chat_id,
            "message_id": message_payload.get("result", {}).get("message_id"),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def _send_mobile_alert(config, system, side, entry, sl, tp, quality, symbol, regime, live_price):
    """Send a formatted pre-execution trade alert."""
    _, chat_id = _telegram_credentials(config)
    if not chat_id:
        logger.warning("Telegram credentials not fully configured. Skipping mobile alert.")
        return

    # Sanitize strings for Telegram Markdown (replace underscores with spaces)
    clean_regime = str(regime).replace("_", " ")
    clean_system = str(system).replace("_", " ")

    msg = (
        "*AQRS V3 TRADE ALERT*\n\n"
        f"• *Symbol:* {symbol}\n"
        f"• *System:* {clean_system} ({quality})\n"
        f"• *Regime:* {clean_regime}\n"
        f"• *Signal:* {side.upper()} @ {entry:.2f}\n"
        f"• *Execution Price:* {live_price:.2f}\n"
        f"• *SL:* {sl:.2f} | *TP:* {tp:.2f}"
    )

    send_telegram_msg(config, msg)
    logger.info("Mobile alert sent: %s %s", system, side.upper())


def run(config, execute: bool = False, live_tick=None): # Added live_tick parameter
    logger.info("Execution Agent: checking for live signals...")
    bridge = MT5Bridge(config)

    if getattr(config, "USE_V2_EXECUTION", False):
        logger.info("V2 execution fallback active.")
        return

    # Initialize MT5 and get live tick data EARLY for freshness check and alerts
    if not bridge.initialize_and_validate():
        return

    bridge.sync_closed_trades()

    trade_setups_path = getattr(config.paths, "trade_setups", Path("data/features/trade_setups.csv"))
    df = pd.read_csv(trade_setups_path, low_memory=False)
    if df.empty:
        return
    last_signal = df.iloc[-1]
    signal_type = last_signal["confirmed_signal"]
    if signal_type not in ["buy", "sell"]:
        logger.info("No active signal found in the latest candle.")
        return

    symbol = config.market.symbol
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol) # This is where 'tick' is properly assigned
    if not symbol_info or tick is None:
        logger.error("Failed to get market data for %s", symbol)
        return

    # Freshness check now that 'tick' is available
    is_fresh, latency, max_latency = _is_signal_fresh(last_signal, tick, config)
    if not is_fresh:
        logger.warning("⚠️ Signal Rejected: Adjusted Latency %ds (Max: %ds). Check if your Broker Time and System Time are out of sync.", int(latency), int(max_latency))
        return

    logger.info("Live Market Snapshot for %s: Bid=%.5f, Ask=%.5f", symbol, tick.bid, tick.ask)

    if not bridge.check_daily_drawdown():
        return

    if not bridge.check_simultaneous_positions():
        return

    last_signal_dict = last_signal.to_dict()
    last_signal_dict["current_tick_price"] = tick.ask if signal_type == "buy" else tick.bid
    last_signal_dict["spread"] = tick.ask - tick.bid

    allowed, system, lot, reason, is_exploratory = ExecutionGate.evaluate_signal(config, last_signal_dict)

    signal_meta = last_signal_dict
    signal_meta["current_zone"] = (
        f"S:{signal_meta.get('support_level', 0):.2f} "
        f"R:{signal_meta.get('resistance_level', 0):.2f}"
    )
    signal_meta.update({"system": system, "is_exploratory": is_exploratory})

    if not allowed and reason in ["COST_TO_REWARD_REJECTION", "SPREAD_REJECTION"]:
        logger.info("[DEMO OVERRIDE] Bypassing %s to test market entry.", reason)
        allowed = True

    if not allowed:
        bridge.log_blocked_trade(signal_meta, reason)
        return

    live_price = tick.ask if signal_type == "buy" else tick.bid
    _send_mobile_alert(
        config,
        system,
        signal_type,
        last_signal["entry_price"],
        last_signal["stop_loss"],
        last_signal["take_profit"],
        last_signal["quality"],
        symbol,
        last_signal.get("market_regime", "UNKNOWN"),
        live_price
    )

    allow_overlap = getattr(config.backtest, "allow_overlapping_positions", False)
    is_blocked, block_reason = bridge.is_candle_already_traded(symbol, last_signal["time"], allow_overlap)
    if is_blocked:
        logger.warning("🚫 Trade blocked: %s", block_reason)
        return

    if execute:
        request = _prepare_request(symbol, signal_type, lot, tick, signal_meta, symbol_info, config, system)
        bridge.execute_order(request, signal_meta)
    else:
        logger.info(
            "[VALIDATION-AUDIT] System: %s | Signal: %s | Lot: %.2f | Reason: %s",
            system,
            signal_type.upper(),
            lot,
            reason,
        )
        logger.info("Preview mode: signal quality recorded.")


def _prepare_request(symbol, signal_type, lot, tick, signal, symbol_info, config, system):
    price = tick.ask if signal_type == "buy" else tick.bid
    type_order = mt5.ORDER_TYPE_BUY if signal_type == "buy" else mt5.ORDER_TYPE_SELL

    # Robust price normalization using symbol's tick size to avoid 10016 errors
    tick_size = symbol_info.trade_tick_size
    
    def _norm(p):
        if p is None or pd.isna(p): return 0.0
        # Normalize to tick size and then round to broker's digits
        return float(round(round(float(p) / tick_size) * tick_size, symbol_info.digits))

    price = _norm(price)
    sl = _norm(signal.get("stop_loss", 0))
    tp = _norm(signal.get("take_profit", 0))

    # Force a more conservative minimum stop level floor (100 points = 1.00 on XAUUSD)
    # Some brokers report 0 but reject trades closer than 50-100 points during volatility.
    stop_level_points = max(float(symbol_info.trade_stops_level), 100.0)
    
    # Add an extra safety buffer: 5 ticks normally, 10 ticks for ALPHA signals
    safety_buffer_ticks = 10 if system == "ALPHA" else 5
    
    # Final minimum distance calculation
    min_dist_price = (stop_level_points * symbol_info.point) + (safety_buffer_ticks * tick_size)

    # Adjust Stop Loss if too close
    if sl != 0 and abs(price - sl) < min_dist_price:
        sl = price - min_dist_price if signal_type == "buy" else price + min_dist_price
        sl = _norm(sl) # Re-normalize after adjustment
        logger.info("🛡️ SL adjusted to meet broker limits: %.5f", sl)

    # Adjust Take Profit if too close
    if tp != 0 and abs(price - tp) < min_dist_price:
        tp = price + min_dist_price if signal_type == "buy" else price - min_dist_price
        tp = _norm(tp) # Re-normalize after adjustment
        logger.info("🛡️ TP adjusted to meet broker limits: %.5f", tp)

    # Final Directional Integrity Check (Logic Safety)
    if signal_type == "buy":
        if sl != 0 and sl >= price: 
            sl = _norm(price - min_dist_price)
        if tp != 0 and tp <= price:
            tp = _norm(price + min_dist_price)
    else:
        if sl != 0 and sl <= price:
            sl = _norm(price + min_dist_price)
        if tp != 0 and tp >= price:
            tp = _norm(price - min_dist_price)

    logger.info("📐 Final Order Parameters for %s %s: Price=%.5f, SL=%.5f, TP=%.5f", 
                symbol, signal_type.upper(), price, sl, tp)

    # Auto-detect filling mode if not explicitly provided
    filling = getattr(config.market, "order_filling", None)
    if filling is None:
        if symbol_info.filling_mode & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
            filling = mt5.ORDER_FILLING_IOC
        elif symbol_info.filling_mode & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
            filling = mt5.ORDER_FILLING_FOK
        else:
            filling = mt5.ORDER_FILLING_RETURN

    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": type_order,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": getattr(config.market, "max_slippage_points", 10_000),
        "magic": getattr(config.market, "magic_number", 202404),
        "comment": f"AQ_{system}_{signal['quality']}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

def _is_signal_fresh(last_signal, tick, config):
    """Checks if the signal is fresh enough based on broker time."""
    # Ensure both timestamps are naive (no timezone) to prevent massive offset errors
    signal_time = pd.to_datetime(last_signal["time"]).replace(tzinfo=None)
    broker_now = pd.to_datetime(tick.time, unit='s').replace(tzinfo=None)
    
    # M5 candle starting at 08:00 closes at 08:05.
    candle_close_time = signal_time + pd.Timedelta(minutes=5)
    raw_latency = (broker_now - candle_close_time).total_seconds()

    # Timezone Robustness: Brokers often have hour-level offsets (e.g. UTC vs UTC+2).
    # If latency is huge (>45 mins), we strip the hour components to find the true delay.
    if abs(raw_latency) > 3600:
        hours_offset = round(raw_latency / 3600)
        latency = abs(raw_latency - (hours_offset * 3600))
        logger.debug("Timezone Offset Detected: %d hours. Raw: %ds, Adjusted: %ds", hours_offset, int(raw_latency), int(latency))
    else:
        latency = abs(raw_latency)
    
    # Add a 60-second "Processing Buffer" to account for pipeline calculation time
    # and polling intervals, so 301s doesn't cause a rejection.
    max_latency = (getattr(config.live, "max_signal_age_minutes", 5) * 60) + 60
    return (latency <= max_latency), latency, max_latency
