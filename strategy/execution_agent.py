import MetaTrader5 as mt5
import pandas as pd
from core.logging_utils import get_logger
from strategy.mt5_bridge import MT5Bridge
from strategy.execution_gate import ExecutionGate

logger = get_logger(__name__)

def run(config, execute: bool = False):
    logger.info("🚀 Execution Agent: checking for live signals...")
    bridge = MT5Bridge(config)

    # 1. Load Data
    df = pd.read_csv(config.paths.trade_setups, low_memory=False)
    if df.empty:
        return

    last_signal = df.iloc[-1]
    signal_type = last_signal["confirmed_signal"]
    
    if signal_type not in ["buy", "sell"]:
        logger.info("⏸️ No active signal found in the latest candle.")
        return

    # 2. Execution Gate Evaluation
    allowed, system, lot, reason, is_exploratory = ExecutionGate.evaluate_signal(config, last_signal)
    
    # Enrich signal data with system decision for logging
    signal_meta = last_signal.to_dict()
    signal_meta.update({"system": system, "is_exploratory": is_exploratory})

    if not allowed:
        bridge.log_blocked_trade(signal_meta, reason)
        return

    # 3. MT5 Bridge Initialization
    if not bridge.initialize_and_validate():
        return

    # 4. Symbol Validation
    symbol = config.market.symbol
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not symbol_info or not tick:
        logger.error("❌ Failed to get market data for %s", symbol)
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
        request = _prepare_request(symbol, signal_type, lot, tick, last_signal, symbol_info, config, system)
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
        "magic": getattr(config.market, "magic_number", 202404),
        "comment": f"AQ_{system}_{signal['quality']}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": getattr(config.market, "order_filling", mt5.ORDER_FILLING_IOC),
    }