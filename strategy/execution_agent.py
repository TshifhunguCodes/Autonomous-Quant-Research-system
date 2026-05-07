from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from core.logging_utils import get_logger
from notifications.telegram_notifier import (
    send_telegram_msg,
    send_trade_alert,
    telegram_credentials,
    test_telegram_connection as _test_telegram_connection,
)
from engines.dynamic_exit_engine import DynamicExitEngine
from strategy.execution_gate import ExecutionGate
from strategy.mt5_bridge import MT5Bridge
from strategy.trade_lifecycle_manager import TradeLifecycleManager

logger = get_logger(__name__)


def test_telegram_connection(config):
    return _test_telegram_connection(config)


def _send_mobile_alert(config, signal_meta, live_price):
    _, chat_id = telegram_credentials(config)
    if not chat_id:
        logger.warning("Telegram credentials not fully configured. Skipping mobile alert.")
        return

    payload = dict(signal_meta)
    payload["price"] = live_price
    payload["exit_state"] = payload.get("exit_state", "OPEN")
    send_trade_alert(config, "ENTRY_ALERT", payload)
    logger.info("Mobile alert sent: %s %s", payload.get("system", "UNKNOWN"), payload.get("confirmed_signal", ""))


def run(config, execute: bool = False, live_tick=None, signal_data=None):
    logger.info("Execution Agent: checking for live signals...")
    bridge = MT5Bridge(config)

    if getattr(config, "USE_V2_EXECUTION", False):
        logger.info("V2 execution fallback active.")
        return

    # Optimization: Use passed signal_data to skip disk I/O
    if signal_data is not None:
        last_signal = signal_data
    else:
        trade_setups_path = getattr(config.paths, "trade_setups", Path("data/features/trade_setups.csv"))
        df = pd.read_csv(trade_setups_path, low_memory=False)
        if df.empty:
            return
        last_signal = df.iloc[-1].to_dict()

    signal_type = last_signal["confirmed_signal"]
    if signal_type not in ["buy", "sell"]:
        logger.info("No active signal found in the latest candle.")
        return

    # Ensure MT5 is connected for tick data
    if not mt5.initialize():
        logger.error("MT5 failed to initialize for signal check.")
        return

    # Resolve symbol with fallback to variants
    from agents.data_agent import resolve_symbol
    try:
        symbol = resolve_symbol(config.market.symbol)
    except RuntimeError as e:
        logger.error("Failed to resolve symbol for execution: %s", e)
        return
    
    symbol_info = mt5.symbol_info(symbol)
    tick = live_tick or mt5.symbol_info_tick(symbol)
    if not symbol_info or tick is None:
        logger.error("Failed to get market data for %s", symbol)
        return

    is_fresh, latency, max_latency = _is_signal_fresh(last_signal, tick, config)
    if not is_fresh:
        logger.warning(
            "Signal Rejected: Adjusted Latency %ds (Max: %ds). Check broker time alignment.",
            int(latency),
            int(max_latency),
        )
        return

    # Prepare metadata for Evaluation
    last_signal_dict = dict(last_signal)
    last_signal_dict["current_tick_price"] = tick.ask if signal_type == "buy" else tick.bid
    last_signal_dict["spread"] = tick.ask - tick.bid
    last_signal_dict["current_time"] = pd.to_datetime(tick.time, unit="s")
    last_signal_dict["symbol"] = symbol
    last_signal_dict.update(
        TradeLifecycleManager.build_trade_plan(
            last_signal_dict,
            signal_type,
            last_signal_dict["current_tick_price"],
            config,
        )
    )

    allowed, system, lot, reason, is_exploratory = ExecutionGate.evaluate_signal(config, last_signal_dict)

    # PRIORITY 1: Send Telegram Alert immediately if the signal is valid
    if allowed:
        live_price = tick.ask if signal_type == "buy" else tick.bid
        alert_meta = {**last_signal_dict, "system": system, "is_exploratory": is_exploratory}
        alert_meta.update(_build_live_trade_levels(alert_meta, signal_type, live_price, config))
        _send_mobile_alert(config, alert_meta, live_price)
        logger.info("🚀 High-Priority Alert dispatched to Telegram.")

    # PRIORITY 2: Heavy lifting and Bridge validation
    if not bridge.initialize_and_validate():
        return

    bridge.sync_closed_trades() # Syncing history is slow, done after alert.

    if not bridge.check_daily_drawdown() or not bridge.check_simultaneous_positions():
        return

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

    # Manage existing positions
    lifecycle_events = TradeLifecycleManager.manage_open_positions(config, signal_meta, symbol_info, tick)
    for event in lifecycle_events:
        send_trade_alert(config, event["alert_type"], event)

    live_price = tick.ask if signal_type == "buy" else tick.bid
    signal_meta.update(_build_live_trade_levels(signal_meta, signal_type, live_price, config))
    signal_meta.update(DynamicExitEngine.build_exit_plan(signal_meta, 0.0, signal_type))

    allow_overlap = getattr(config.backtest, "allow_overlapping_positions", False)
    is_blocked, block_reason = bridge.is_candle_already_traded(symbol, last_signal["time"], allow_overlap)
    if is_blocked:
        logger.warning("Trade blocked: %s", block_reason)
        return
    if system == "FLOW_EXP":
        limits_ok, limits_reason = _flow_position_limits_ok(config, signal_meta)
        if not limits_ok:
            logger.warning("Trade blocked: %s", limits_reason)
            bridge.log_blocked_trade(signal_meta, limits_reason)
            return
    if not TradeLifecycleManager.can_stack(signal_meta, tick):
        logger.warning("Trade blocked: stacking only allowed during SCALE_ALLOWED.")
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
    tick_size = symbol_info.trade_tick_size

    def _norm(p):
        if p is None or pd.isna(p):
            return 0.0
        return float(round(round(float(p) / tick_size) * tick_size, symbol_info.digits))

    price = _norm(price)
    signal.update(_build_live_trade_levels(signal, signal_type, price, config))
    sl = _norm(signal.get("stop_loss", 0))
    tp = _norm(signal.get("take_profit", 0))

    stop_level_points = max(float(symbol_info.trade_stops_level), 100.0)
    safety_buffer_ticks = 10 if system == "ALPHA" else 5
    min_dist_price = (stop_level_points * symbol_info.point) + (safety_buffer_ticks * tick_size)

    if sl != 0 and abs(price - sl) < min_dist_price:
        sl = _norm(price - min_dist_price if signal_type == "buy" else price + min_dist_price)
        logger.info("SL adjusted to meet broker limits: %.5f", sl)

    if tp != 0 and abs(price - tp) < min_dist_price:
        tp = _norm(price + min_dist_price if signal_type == "buy" else price - min_dist_price)
        logger.info("TP adjusted to meet broker limits: %.5f", tp)

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

    logger.info(
        "Final Order Parameters for %s %s: Price=%.5f, SL=%.5f, TP=%.5f",
        symbol,
        signal_type.upper(),
        price,
        sl,
        tp,
    )

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
        "comment": _build_order_comment(system, signal),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }


def _build_live_trade_levels(signal, signal_type, live_price, config):
    plan = TradeLifecycleManager.build_trade_plan(signal, signal_type, live_price, config)
    if str(signal.get("signal", "")).upper() != "FLOW" and str(signal.get("system", "")) != "FLOW_EXP":
        return plan

    atr_value = float(signal.get("atr14", signal.get("atr", 0.0)) or 0.0)
    if atr_value <= 0:
        return plan

    flow_type = str(signal.get("flow_trade_type", "MOMENTUM_CONTINUATION"))
    multiplier = float(signal.get("flow_atr_sl_multiplier", _flow_sl_multiplier(flow_type)) or _flow_sl_multiplier(flow_type))
    rr_ratio = max(1.5, float(signal.get("flow_rr_ratio", 2.0) or 2.0))
    spread = float(signal.get("spread", 0.0) or 0.0)
    stop_distance = max(atr_value * multiplier, spread * 2.0)

    if signal_type == "buy":
        stop_loss = live_price - stop_distance
        take_profit = live_price + (stop_distance * rr_ratio)
    else:
        stop_loss = live_price + stop_distance
        take_profit = live_price - (stop_distance * rr_ratio)

    price_drift = abs(live_price - float(signal.get("entry_price", live_price)))
    return {
        **plan,
        "entry_price": float(live_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "stop_distance": float(stop_distance),
        "flow_rr_ratio": float(rr_ratio),
        "flow_atr_sl_multiplier": float(multiplier),
        "smart_stop_ok": stop_distance >= max(spread * 2.0, 0.0),
        "price_drift_ok": price_drift <= max(spread * 10.0, atr_value * 1.5),
        "price_drift": float(price_drift),
        "max_price_drift": float(max(spread * 10.0, atr_value * 1.5)),
    }


def _flow_sl_multiplier(flow_type):
    return {
        "EXHAUSTION_FADE": 0.3,
        "MICRO_RETRACEMENT_REENTRY": 0.4,
        "MOMENTUM_CONTINUATION": 0.5,
        "EARLY_REVERSAL_ENTRY": 0.6,
    }.get(str(flow_type), 0.5)


def _flow_position_limits_ok(config, signal):
    positions = mt5.positions_get(symbol=config.market.symbol) or []
    flow_positions = [p for p in positions if "FLOW_EXP" in str(getattr(p, "comment", ""))]
    alpha_positions = [p for p in positions if "ALPHA" in str(getattr(p, "comment", ""))]
    alpha_in_drawdown = any(float(getattr(p, "profit", 0.0) or 0.0) < 0 for p in alpha_positions)
    max_flow = 1 if alpha_in_drawdown else int(signal.get("flow_max_open_trades", 3) or 3)

    if len(flow_positions) >= max_flow:
        return False, "FLOW_MAX_OPEN_TRADES_REACHED"

    flow_type = str(signal.get("flow_trade_type", "NONE"))
    if flow_type == "EXHAUSTION_FADE" and any("EXH" in str(getattr(p, "comment", "")) for p in flow_positions):
        return False, "FLOW_EXHAUSTION_FADE_ALREADY_ACTIVE"
    if flow_type == "MICRO_RETRACEMENT_REENTRY" and any("REENT" in str(getattr(p, "comment", "")) for p in flow_positions):
        return False, "FLOW_REENTRY_ALREADY_ACTIVE"
    return True, ""


def _build_order_comment(system, signal):
    quality = str(signal.get("quality", "NA"))[:5]
    if system != "FLOW_EXP":
        return f"AQ_{system}_{quality}"
    flow_code = {
        "MOMENTUM_CONTINUATION": "MOM",
        "MICRO_RETRACEMENT_REENTRY": "REENT",
        "EXHAUSTION_FADE": "EXH",
        "EARLY_REVERSAL_ENTRY": "REV",
    }.get(str(signal.get("flow_trade_type", "FLOW")), "FLOW")
    return f"AQ_FLOW_EXP_{flow_code}_{quality}"


def _is_signal_fresh(last_signal, tick, config):
    signal_time = pd.to_datetime(last_signal["time"]).replace(tzinfo=None)
    broker_now = pd.to_datetime(tick.time, unit="s").replace(tzinfo=None)
    # A signal generated at 10:00 (M5) is for the candle covering 10:00-10:05.
    # The earliest it can be processed is 10:05:00.
    candle_close_time = signal_time + pd.Timedelta(minutes=5) 
    raw_latency = (broker_now - candle_close_time).total_seconds()

    # Timezone Offset Correction:
    # If the broker server time and signal data timestamps have a whole-hour mismatch
    # (common when mixing UTC history with local server ticks), we adjust for it.
    if abs(raw_latency) > 1800: # Over 30 mins difference suggests timezone offset
        hours_offset = round(raw_latency / 3600)
        latency = abs(raw_latency - (hours_offset * 3600))
    else:
        latency = max(0.0, raw_latency)

    is_flow = str(last_signal.get("signal", "")).upper() == "FLOW" or str(last_signal.get("flow_trade_type", "NONE")) != "NONE"
    # We allow 5 minutes for FLOW and 10 minutes for ALPHA.
    # If your latency was 291s, this will now PASS (291 < 300).
    max_latency = (5 if is_flow else 10) * 60 
    return (latency <= max_latency), latency, max_latency
