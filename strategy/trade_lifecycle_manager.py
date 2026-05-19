from __future__ import annotations

from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from core.logging_utils import get_logger
from engines.dynamic_exit_engine import DynamicExitEngine

logger = get_logger(__name__)


class TradeLifecycleManager:
    """Lightweight post-entry management and smart stop placement."""

    OPEN = "OPEN"
    PROTECTED = "PROTECTED"
    SCALE_ALLOWED = "SCALE_ALLOWED"
    WEAKENING = "WEAKENING"
    EXIT_WARNING = "EXIT_WARNING"
    FORCE_EXIT = "FORCE_EXIT"
    _partial_exit_tickets: set[int] = set()

    @staticmethod
    def _truthy(value: Any) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def build_trade_plan(signal: dict[str, Any], signal_type: str, live_price: float, config: Any) -> dict[str, Any]:
        atr_value = float(signal.get("atr14", signal.get("atr", 0.0)) or 0.0)
        spread = float(signal.get("spread", 0.0) or 0.0)
        signal_name = str(signal.get("signal", "FLOW")).upper()
        rr_ratio = float(getattr(config.risk, "rr_ratio", 2.0))
        lifecycle_state = str(signal.get("lifecycle_state", "TREND_HEALTHY"))

        # Wider ATR multipliers to avoid invalid stops (error 10016)
        # ALPHA: 2.8x ATR, FLOW: 2.2x ATR - more breathing room
        atr_multiplier = 2.8 if signal_name == "ALPHA" else 2.2
        atr_stop = atr_value * atr_multiplier if atr_value > 0 else 0.0

        candle_low = float(signal.get("low", live_price))
        candle_high = float(signal.get("high", live_price))
        wick_noise = max(live_price - candle_low, candle_high - live_price, spread * 2.0)

        if signal_type == "buy":
            structural_anchor = min(
                float(signal.get("last_swing_low", candle_low) or candle_low),
                float(signal.get("support_level", candle_low) or candle_low),
                candle_low,
            )
            if TradeLifecycleManager._truthy(signal.get("order_block", 0)) or TradeLifecycleManager._truthy(signal.get("demand_zone", 0)):
                structural_anchor = min(structural_anchor, float(signal.get("open", candle_low) or candle_low), candle_low)
        else:
            structural_anchor = max(
                float(signal.get("last_swing_high", candle_high) or candle_high),
                float(signal.get("resistance_level", candle_high) or candle_high),
                candle_high,
            )
            if TradeLifecycleManager._truthy(signal.get("supply_zone", 0)) or TradeLifecycleManager._truthy(signal.get("order_block", 0)):
                structural_anchor = max(structural_anchor, float(signal.get("open", candle_high) or candle_high), candle_high)

        volatility_buffer = 1.0
        if lifecycle_state in ["BREAKOUT_EXPANSION", "TREND_EXHAUSTING"] or TradeLifecycleManager._truthy(signal.get("volatility", 0)):
            volatility_buffer = 1.35  # Slightly higher for volatile conditions

        # Wider floor distance - minimum 12 points instead of 8, better spread coverage
        floor_distance = max(atr_stop, spread * 3.5, wick_noise * 1.2, 12.0) * volatility_buffer
        structure_buffer = max(spread * 2.5, atr_value * 0.35 if atr_value > 0 else 3.0)

        if signal_type == "buy":
            structural_stop = structural_anchor - structure_buffer
            stop_loss = min(live_price - floor_distance, structural_stop)
            stop_distance = live_price - stop_loss
            take_profit = live_price + (stop_distance * rr_ratio)
        else:
            structural_stop = structural_anchor + structure_buffer
            stop_loss = max(live_price + floor_distance, structural_stop)
            stop_distance = stop_loss - live_price
            take_profit = live_price - (stop_distance * rr_ratio)

        price_drift = abs(live_price - float(signal.get("entry_price", live_price)))
        max_drift = max(spread * 4.0, atr_value * 0.60 if atr_value > 0 else 10.0)
        unrealistic_stop = stop_distance < max(spread * 3.0, atr_value * 0.75 if atr_value > 0 else 8.0, wick_noise)

        return {
            "entry_price": float(live_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "stop_distance": float(stop_distance),
            "smart_stop_ok": not unrealistic_stop,
            "price_drift_ok": price_drift <= max_drift,
            "price_drift": float(price_drift),
            "max_price_drift": float(max_drift),
            "structure_stop_anchor": float(structural_anchor),
        }

    @staticmethod
    def classify_trade_state(position: Any, signal: dict[str, Any], tick: Any) -> str:
        entry = float(position.price_open)
        side = "buy" if position.type == mt5.ORDER_TYPE_BUY else "sell"
        current_price = float(tick.bid if side == "buy" else tick.ask)
        stop_loss = float(position.sl or signal.get("stop_loss", entry))
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return TradeLifecycleManager.OPEN

        if side == "buy":
            unrealized_r = (current_price - entry) / stop_distance
        else:
            unrealized_r = (entry - current_price) / stop_distance
        return DynamicExitEngine.build_exit_plan(signal, unrealized_r, side)["exit_state"]

    @staticmethod
    def manage_open_positions(config: Any, signal: dict[str, Any], symbol_info: Any, tick: Any) -> list[dict[str, Any]]:
        positions = mt5.positions_get(symbol=config.market.symbol) or []
        if not positions:
            return []

        events: list[dict[str, Any]] = []

        for position in positions:
            state = TradeLifecycleManager.classify_trade_state(position, signal, tick)
            stop_loss = float(position.sl or signal.get("stop_loss", 0.0))
            stop_distance = abs(float(position.price_open) - stop_loss)
            current_price = float(tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask)
            if stop_distance > 0:
                unrealized_r = (
                    (current_price - float(position.price_open)) / stop_distance
                    if position.type == mt5.ORDER_TYPE_BUY
                    else (float(position.price_open) - current_price) / stop_distance
                )
            else:
                unrealized_r = 0.0
            exit_plan = DynamicExitEngine.build_exit_plan(
                signal,
                unrealized_r,
                "buy" if position.type == mt5.ORDER_TYPE_BUY else "sell",
            )
            base_event = {
                **signal,
                "ticket": int(position.ticket),
                "symbol": position.symbol,
                "price": current_price,
                "entry_price": float(position.price_open),
                "stop_loss": stop_loss,
                "take_profit": float(position.tp or signal.get("take_profit", 0.0)),
                "confirmed_signal": "buy" if position.type == mt5.ORDER_TYPE_BUY else "sell",
                **exit_plan,
            }
            if state == TradeLifecycleManager.PROTECTED:
                moved = TradeLifecycleManager._move_to_break_even(position, symbol_info, tick)
                if moved:
                    events.append({"alert_type": "CONTINUATION_ALERT", **base_event})
            elif state == TradeLifecycleManager.SCALE_ALLOWED:
                events.append({"alert_type": "CONTINUATION_ALERT", **base_event})
            elif state in [TradeLifecycleManager.WEAKENING, TradeLifecycleManager.EXIT_WARNING]:
                tightened = TradeLifecycleManager._tighten_stop(position, signal, symbol_info, tick, exit_plan)
                if tightened:
                    alert_type = "REVERSAL_WARNING_ALERT" if state == TradeLifecycleManager.EXIT_WARNING else "WEAKENING_ALERT"
                    events.append({"alert_type": alert_type, **base_event})
                if exit_plan["partial_taken"]:
                    partial = TradeLifecycleManager._partial_exit_position(position, symbol_info, tick)
                    if partial:
                        events.append({"alert_type": "PARTIAL_EXIT_ALERT", **base_event})
            elif state == TradeLifecycleManager.FORCE_EXIT:
                closed = TradeLifecycleManager._close_position(position, symbol_info, tick)
                if closed:
                    events.append({"alert_type": "FORCE_EXIT_ALERT", **base_event})
        return events

    @staticmethod
    def can_stack(signal: dict[str, Any], tick: Any) -> bool:
        positions = mt5.positions_get(symbol=signal.get("symbol")) or []
        if not positions:
            return True
        for position in positions:
            state = TradeLifecycleManager.classify_trade_state(position, signal, tick)
            if state != TradeLifecycleManager.SCALE_ALLOWED:
                return False
        return True

    @staticmethod
    def _move_to_break_even(position: Any, symbol_info: Any, tick: Any) -> bool:
        tick_size = symbol_info.trade_tick_size
        be_buffer = max(symbol_info.point * 5, tick_size * 3)
        new_sl = position.price_open + be_buffer if position.type == mt5.ORDER_TYPE_BUY else position.price_open - be_buffer
        current_sl = float(position.sl or 0.0)
        if current_sl and ((position.type == mt5.ORDER_TYPE_BUY and current_sl >= new_sl) or (position.type != mt5.ORDER_TYPE_BUY and current_sl <= new_sl)):
            return False
        return TradeLifecycleManager._modify_position_sl(position, new_sl)

    @staticmethod
    def _tighten_stop(position: Any, signal: dict[str, Any], symbol_info: Any, tick: Any, exit_plan: dict[str, Any] | None = None) -> bool:
        atr_value = float(signal.get("atr14", signal.get("atr", 0.0)) or 0.0)
        if atr_value <= 0:
            return False
        tick_size = symbol_info.trade_tick_size
        current_price = float(tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask)
        planned_distance = float((exit_plan or {}).get("dynamic_trailing_distance", 0.0) or 0.0)
        tighten_distance = max(planned_distance, atr_value * 0.8, symbol_info.point * 20, tick_size * 5)
        new_sl = current_price - tighten_distance if position.type == mt5.ORDER_TYPE_BUY else current_price + tighten_distance
        current_sl = float(position.sl or 0.0)
        if current_sl and ((position.type == mt5.ORDER_TYPE_BUY and new_sl <= current_sl) or (position.type != mt5.ORDER_TYPE_BUY and new_sl >= current_sl)):
            return False
        return TradeLifecycleManager._modify_position_sl(position, new_sl)

    @staticmethod
    def _modify_position_sl(position: Any, new_sl: float) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": float(new_sl),
            "tp": float(position.tp or 0.0),
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Trade lifecycle updated SL for ticket %s", position.ticket)
            return True
        return False

    @staticmethod
    def _partial_exit_position(position: Any, symbol_info: Any, tick: Any) -> bool:
        if int(position.ticket) in TradeLifecycleManager._partial_exit_tickets:
            return False

        min_volume = float(getattr(symbol_info, "volume_min", 0.01) or 0.01)
        step = float(getattr(symbol_info, "volume_step", min_volume) or min_volume)
        partial_volume = max(min_volume, round((float(position.volume) * 0.5) / step) * step)
        if partial_volume >= float(position.volume):
            return False

        close_price = float(tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": partial_volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": close_price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "AQRS_PARTIAL_EXIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            TradeLifecycleManager._partial_exit_tickets.add(int(position.ticket))
            logger.info("Trade lifecycle partial-exit closed %.2f on ticket %s", partial_volume, position.ticket)
            return True
        return False

    @staticmethod
    def _close_position(position: Any, symbol_info: Any, tick: Any) -> bool:
        close_price = float(tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": close_price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "AQRS_FORCE_EXIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Trade lifecycle force-exit closed ticket %s", position.ticket)
            return True
        return False
