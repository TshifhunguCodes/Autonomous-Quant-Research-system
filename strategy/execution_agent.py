import MetaTrader5 as mt5
import pandas as pd
import requests
from core.logging_utils import get_logger
from pathlib import Path
from strategy.mt5_bridge import MT5Bridge
from strategy.execution_gate import ExecutionGate

logger = get_logger(__name__)

def send_telegram_msg(config, text):
    """Generic helper to send any text message to Telegram."""
    token = getattr(config, "TELEGRAM_TOKEN", "8728107155:AAErj5Vo7rYsFkrTzD0nKtFh0JNC3_q3Wek")
    chat_id = getattr(config, "TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        logger.error(f"❌ Telegram Error: {e}")

def _send_mobile_alert(config, system, side, entry, sl, tp, quality, symbol, regime):
    """Formatted trade alerts."""
    if getattr(config, "TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE") == "YOUR_CHAT_ID_HERE":
        logger.warning("Telegram credentials not fully configured. Skipping mobile alert.")
        return

    msg = (
        f"🚀 *AQRS V3 TRADE ALERT*\n\n"
        f"📈 **Symbol:** {symbol}\n"
        f"🎯 **System:** {system} ({quality})\n"
        f"🔭 **Regime:** {regime}\n"
        f"⚡ **Signal:** {side.upper()} @ {entry:.5f}\n"
        f"🛑 **SL:** {sl:.5f} | ✅ **TP:** {tp:.5f}"
    )

    send_telegram_msg(config, msg)
    logger.info(f"📱 Mobile Alert Sent: {system} {side.upper()}")

def run(config, execute: bool = False):
    logger.info("🚀 Execution Agent: checking for live signals...")
    bridge = MT5Bridge(config)

    # Task 6: Fallback Switch
    use_v2 = getattr(config, "USE_V2_EXECUTION", False)
    if use_v2:
        logger.info("⚠️ V2 Execution Fallback Active. Routing to V2 emitter logic...")
        return

    # Task 1: Adaptive Learning - Sync Outcomes
    bridge.sync_closed_trades()

    # 1. Load Data
    trade_setups_path = getattr(config.paths, "trade_setups", Path("data/features/trade_setups.csv"))
    df = pd.read_csv(trade_setups_path, low_memory=False)
    if df.empty:
        return

    last_signal = df.iloc[-1]
    signal_type = last_signal["confirmed_signal"]
    
    if signal_type not in ["buy", "sell"]:
        logger.info("⏸️ No active signal found in the latest candle.")
        return

    # 3. MT5 Bridge Initialization (Moved up for real-time validation)
    if not bridge.initialize_and_validate():
        return

    # 4. Symbol Validation & Ticks
    symbol = config.market.symbol
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not symbol_info or tick is None:
        logger.error("❌ Failed to get market data for %s", symbol)
        return

    logger.info("📊 Live Market Snapshot for %s: Bid=%.5f, Ask=%.5f", symbol, tick.bid, tick.ask)

    # Task 4: Hard Risk Controls
    if not bridge.check_daily_drawdown():
        return
        
    if not bridge.check_simultaneous_positions():
        return

    # 2. V3 Decision Engine / Execution Gate Evaluation
    # Pass current tick price for Slippage Guard (Task 4)
    last_signal_dict = last_signal.to_dict()
    last_signal_dict["current_tick_price"] = tick.ask if signal_type == "buy" else tick.bid

    allowed, system, lot, reason, is_exploratory = ExecutionGate.evaluate_signal(config, last_signal_dict)
    
    # Enrich signal data with system decision for logging
    signal_meta = last_signal_dict
    # Task 5 Logging fields enrichment: Zone Type & Metadata
    signal_meta["current_zone"] = f"S:{signal_meta.get('support_level', 0):.2f} R:{signal_meta.get('resistance_level', 0):.2f}"
    signal_meta.update({"system": system, "is_exploratory": is_exploratory})

    # Send Alert BEFORE execution
    if allowed:
        _send_mobile_alert(
            config, system, signal_type, 
            last_signal["entry_price"], last_signal["stop_loss"], 
            last_signal["take_profit"], last_signal["quality"],
            symbol, last_signal.get("market_regime", "UNKNOWN")
        )

    if not allowed:
        bridge.log_blocked_trade(signal_meta, reason)
        return

    # 5. One-Trade-Per-Candle Sync Gate
    if bridge.is_candle_already_traded(symbol, last_signal["time"]):
        logger.warning("🚫 Trade Blocked: Position already exists for current candle %s", last_signal["time"])
        return

    # 6. Real-time Spread Check
    max_spread = getattr(config.market, "max_spread_allowed", 30)
    if (tick.ask - tick.bid) > (max_spread * symbol_info.point):
        bridge.log_blocked_trade(signal_meta, "SPREAD_TOO_WIDE")
        return

    # 7. Execution
    if execute:
        request = _prepare_request(symbol, signal_type, lot, tick, signal_meta, symbol_info, config, system)
        bridge.execute_order(request, signal_meta)
    else:
        logger.info("🔍 [VALIDATION-AUDIT] System: %s | Signal: %s | Lot: %.2f | Reason: %s", 
                    system, signal_type.upper(), lot, reason)
        logger.info("✅ Preview mode: Signal quality recorded.")

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
        "deviation": getattr(config.market, "max_slippage_points", 10), # Task 4: Slippage Guard
        "magic": getattr(config.market, "magic_number", 202404),
        "comment": f"AQ_{system}_{signal['quality']}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": getattr(config.market, "order_filling", mt5.ORDER_FILLING_IOC),
    }