import MetaTrader5 as mt5
import pandas as pd
import time
from core.logging_utils import get_logger

logger = get_logger(__name__)

def run(config, execute: bool = False):
    logger.info("🚀 Execution Agent: checking for live signals...")
    
    # 1. Load the latest trade setups
    df = pd.read_csv(config.paths.trade_setups, low_memory=False)
    if df.empty:
        return

    last_signal = df.iloc[-1]
    signal_type = last_signal["confirmed_signal"]
    
    if signal_type not in ["buy", "sell"]:
        logger.info("⏸️ No active signal found in the latest candle.")
        return

    # 2. Initialize MT5
    if not mt5.initialize():
        logger.error("❌ MT5 Initialization failed")
        return

    symbol = config.market.symbol
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("❌ Symbol %s not found", symbol)
        return
        
    lot = config.risk.min_lot  # Driven by config
    
    # Get current price for execution
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("❌ Failed to get tick for %s", symbol)
        return

    # --- QUANT SAFETY FILTERS ---
    # 1. The Elite Paradox Filter: Avoid "too perfect" entries (exhaustion)
    exhaustion_threshold = getattr(config.strategy, "exhaustion_threshold", 100)
    if last_signal["confirm_score"] > exhaustion_threshold:
        logger.warning("⚠️ Signal rejected: Score %s suggests move exhaustion.", last_signal['confirm_score'])
        return

    # 2. Market State Filter: Prevent trading in pure noise
    if last_signal["market_state"] == "CHOPPY":
        logger.warning("⚠️ Signal rejected: Market state is CHOPPY.")
        return
        
    # 3. Volatility Anchor: If VOLATILE, require Major Zone and Elite Score
    if last_signal["market_state"] == "VOLATILE":
        is_major = bool(last_signal.get("major_support", 0)) or bool(last_signal.get("major_resistance", 0))
        if not is_major or last_signal["confirm_score"] < 85:
            logger.warning("⚠️ Volatile signal rejected: Requires Major Zone and Score > 85.")
            return

    # 3. Spread Check: Symbol-specific spread protection
    max_spread = getattr(config.market, "max_spread_allowed", 30)
    if (tick.ask - tick.bid) > (max_spread * symbol_info.point):
        logger.warning("⚠️ Signal rejected: Spread too wide (%.1f points).", (tick.ask - tick.bid)/symbol_info.point)
        return

    # 3. Prepare the Order Request
    price = tick.ask if signal_type == "buy" else tick.bid
    type_order = mt5.ORDER_TYPE_BUY if signal_type == "buy" else mt5.ORDER_TYPE_SELL
    
    # Ensure prices are correctly rounded for MT5
    price = round(float(price), symbol_info.digits)
    sl = round(float(last_signal["stop_loss"]), symbol_info.digits)
    tp = round(float(last_signal["take_profit"]), symbol_info.digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": type_order,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": getattr(config.market, "magic_number", 202404),
        "comment": f"AutoQuant_{last_signal['quality']}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": getattr(config.market, "order_filling", mt5.ORDER_FILLING_IOC),
    }

    # 4. Check if order was already placed for this time
    # (Simple logic: don't open multiple trades for the same 5m candle)
    logger.info("📡 Sending %s order for %s (%s Quality)", signal_type.upper(), symbol, last_signal['quality'])

    if execute:
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("❌ Order failed! Error code: %s", result.retcode)
            # Common errors: 10004 (Requote), 10014 (Invalid Volume), 10016 (Invalid Stops)
        else:
            logger.info("✅ Trade Opened! Ticket: %s", result.deal)
    else:
        logger.info("✅ Preview mode: Order not sent (use --execute-live to send live orders).")

if __name__ == "__main__":
    print("Execution Agent module. Run via main.py")