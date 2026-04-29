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

    print(f"Attempting Telegram message to {chat_id}...")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
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


def _send_mobile_alert(config, system, side, entry, sl, tp, quality, symbol, regime):
    """Send a formatted pre-execution trade alert."""
    _, chat_id = _telegram_credentials(config)
    if not chat_id:
        logger.warning("Telegram credentials not fully configured. Skipping mobile alert.")
        return

    msg = (
        "*AQRS V3 TRADE ALERT - BEFORE EXECUTION*\n\n"
        f"**Symbol:** {symbol}\n"
        f"**System:** {system} ({quality})\n"
        f"**Regime:** {regime}\n"
        f"**Signal:** {side.upper()} @ {entry:.5f}\n"
        f"**SL:** {sl:.5f} | **TP:** {tp:.5f}"
    )

    send_telegram_msg(config, msg)
    logger.info("Mobile alert sent: %s %s", system, side.upper())


def run(config, execute: bool = False):
    logger.info("Execution Agent: checking for live signals...")
    bridge = MT5Bridge(config)

    if getattr(config, "USE_V2_EXECUTION", False):
        logger.info("V2 execution fallback active.")
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

    if not bridge.initialize_and_validate():
        return

    symbol = config.market.symbol
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not symbol_info or tick is None:
        logger.error("Failed to get market data for %s", symbol)
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

    if bridge.is_candle_already_traded(symbol, last_signal["time"]):
        logger.warning("Trade blocked: position already exists for current candle %s", last_signal["time"])
        return

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
    )

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

    price = round(float(price), symbol_info.digits)
    sl = round(float(signal["stop_loss"]), symbol_info.digits)
    tp = round(float(signal["take_profit"]), symbol_info.digits)

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
        "type_filling": getattr(config.market, "order_filling", mt5.ORDER_FILLING_IOC),
    }
