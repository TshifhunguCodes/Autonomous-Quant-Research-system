import MetaTrader5 as mt5
import pandas as pd
import time
from core.logging_utils import get_logger

logger = get_logger(__name__)

def _get_live_performance_score(csv_path, lookback):
    try:
        if not csv_path.exists():
            return 1.0
        df = pd.read_csv(csv_path).tail(lookback)
        if df.empty:
            return 1.0
        wins = (df["result"] == "WIN").sum()
        gp = df.loc[df["pnl"] > 0, "pnl"].sum()
        gl = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 1.5
        return (wins / len(df)) + (min(pf, 3.0) / 3.0)
    except Exception:
        return 1.0

def run(config, execute: bool = False):
    logger.info("🚀 Execution Agent: checking for live signals...")
    
    # 1. Load the latest trade setups
    df = pd.read_csv(config.paths.trade_setups, low_memory=False)
    if df.empty:
        return

    last_signal = df.iloc[-1]
    signal_type = last_signal["confirmed_signal"]
    hour = pd.to_datetime(last_signal["time"]).hour
    
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
        
    # --- SESSION & SETUP ADAPTATION ---
    state = last_signal["market_state"]
    quality = last_signal["quality"]
    score = last_signal["confirm_score"]
    setup_key = f"{last_signal['setup']}_{quality}_{state}"

    # 1. Position Sizing with Volatility Adaptation
    lot = config.live.lot
    
    # STRICT ALPHA Eligibility with Regime-based floor
    alpha_eligible = quality == "ELITE" and hour in config.regime.alpha_session_hours and state not in ["CHOPPY", "VOLATILE"]
    if alpha_eligible and state == "TRENDING" and score < 85:
        alpha_eligible = False

    # ADAPTIVE FLOW Eligibility (Always active for data generation)
    flow_eligible = True

    if alpha_eligible:
        is_alpha = True
        logger.info("💎 Mode: ALPHA selected.")
    else:
        is_alpha = False
        logger.info("🌊 Mode: FLOW_EXPLORATORY selected.")
        # Risk scaling for exploratory engine: base lot × flow risk multiplier × 0.5 dampener
        lot = max(0.01, lot * config.regime.flow_risk_multiplier * 0.5)
        logger.info("🛡️ Exploratory risk scaling applied: %.2f lots", lot)

    # 2. Connection and Symbol Validation
    # Get current price for execution
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("❌ Failed to get tick for %s", symbol)
        return

    # --- QUANT SAFETY FILTERS ---
    # 1. The Elite Paradox Filter: Avoid "too perfect" entries (exhaustion)
    exhaustion_threshold = 100
    if last_signal["confirm_score"] > exhaustion_threshold:
        logger.warning("⚠️ Signal rejected: Score %s suggests move exhaustion.", last_signal['confirm_score'])
        return

    # 2. Volatility Adaptation: Low Volatility (CHOPPY)
    if is_alpha and last_signal["market_state"] == "CHOPPY":
        if last_signal["quality"] != "ELITE" or last_signal["confirm_score"] < 80:
            logger.warning("⚠️ Low Volatility rejection: Requires ELITE quality and Score > 80.")
            return

    # 3. Session Hardening: New York Guard (Adaptive)
    if is_alpha and config.regime.adaptive_ny_guard and 13 <= hour <= 20:
        if score < 75:
            logger.warning("⚠️ New York rejection: Requires higher conviction floor (75).")
            return

    # 3. Volatility Adaptation: High Volatility (VOLATILE)
    if is_alpha and last_signal["market_state"] == "VOLATILE":
        is_major = bool(last_signal.get("major_support", 0)) or bool(last_signal.get("major_resistance", 0))
        if not is_major or last_signal["confirm_score"] < 90:
            logger.warning("⚠️ High Volatility rejection: Requires Major Zone and Score > 90.")
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
        "comment": f"AQ_{'ALPHA' if is_alpha else 'FLOW_EXP'}_{last_signal['quality']}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": getattr(config.market, "order_filling", mt5.ORDER_FILLING_IOC),
    }

    # 4. Check if order was already placed for this time
    # Strict enforcement: query existing positions to prevent double execution on the same bar
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            # Check if existing position was opened within the current M5 window
            pos_time = pd.to_datetime(pos.time, unit='s')
            current_candle_time = pd.to_datetime(last_signal["time"])
            if pos_time >= current_candle_time:
                logger.warning("🚫 Trade Blocked: Position already exists for current candle %s", last_signal["time"])
                return

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