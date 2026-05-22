"""
Trailing Stop Manager — Dynamic trailing stops for open positions
Adapted from ATS_US30_NAS into AQRS

Features:
- 50-point trailing stop on profitable positions
- Break-even stop after price moves 1x ATR in our favor
- Lock-in partial profits at 2x ATR
"""
import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """
    Manages trailing stops and break-even for open positions.
    
    Strategy:
    1. Initial stop is set by TradeLifecycleManager
    2. When price moves 1x ATR in our favor → move SL to break-even
    3. When price moves 1.5x ATR in our favor → trail by 50 points
    4. Every tick/poll cycle: check if SL should be tightened
    """
    
    def __init__(self, config=None):
        self.config = config
        self.trail_points = None  # Will be set based on symbol
    
    def manage_trailing_stops(self):
        """
        Check all open positions and update trailing stops if needed.
        Called every polling cycle in live mode.
        
        Returns:
            list of dicts with stop adjustment actions taken
        """
        positions = mt5.positions_get()
        if not positions:
            return []
        
        actions = []
        for pos in positions:
            if not self._is_managed_position(pos):
                continue
            action = self._evaluate_position(pos)
            if action:
                actions.append(action)
        
        return actions

    def _is_managed_position(self, position) -> bool:
        if self.config is None:
            return True
        configured_symbol = str(getattr(self.config.market, "symbol", "") or "").upper()
        position_symbol = str(getattr(position, "symbol", "") or "").upper()
        comment = str(getattr(position, "comment", "") or "").upper()
        magic = getattr(position, "magic", None)
        system_magic = int(getattr(self.config.market, "magic_number", 202404))

        symbol_match = (
            configured_symbol
            and (
                configured_symbol == position_symbol
                or configured_symbol in position_symbol
                or (configured_symbol in {"XAUUSD", "GOLD"} and ("XAU" in position_symbol or "GOLD" in position_symbol))
            )
        )
        try:
            magic_match = magic is not None and int(magic) == system_magic
        except (TypeError, ValueError):
            magic_match = False
        return bool(symbol_match and (comment.startswith("AQ_") or magic_match))
    
    def _evaluate_position(self, position) -> dict:
        """
        Evaluate a single position and decide if trailing stop should be adjusted.
        
        Args:
            position: MT5 position object
        
        Returns:
            dict with action taken (or None if no action needed)
        """
        symbol = position.symbol
        order_type = position.type  # 0=BUY, 1=SELL
        current_sl = position.sl
        current_tp = position.tp
        open_price = position.price_open
        current_price = position.price_current
        
        # Get symbol info for price normalization
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return None
        
        point = symbol_info.point
        digits = symbol_info.digits
        
        # Calculate profit in points
        if order_type == mt5.ORDER_TYPE_BUY:
            profit_points = (current_price - open_price) / point
        else:
            profit_points = (open_price - current_price) / point
        
        # Determine trail distance in points
        trail_points = self._get_trail_points(symbol)
        
        # --- Stage 1: No profit yet → keep original SL ---
        if profit_points <= 0:
            return None
        
        # --- Stage 2: Profit >= 1x trail distance → move to break-even ---
        if profit_points >= trail_points and current_sl == 0:
            # No stop set, or original stop is before break-even
            return None  # Skip if no existing SL to modify
        
        if profit_points >= trail_points:
            # Calculate new SL at break-even + small buffer
            if order_type == mt5.ORDER_TYPE_BUY:
                new_sl = open_price + (trail_points * 0.1 * point)  # 10% buffer above break-even
            else:
                new_sl = open_price - (trail_points * 0.1 * point)
            
            new_sl = round(new_sl, digits)
            
            # Only modify if new SL is better than current
            if order_type == mt5.ORDER_TYPE_BUY and (current_sl is None or new_sl > current_sl):
                return self._modify_stop(position, new_sl, current_tp, "break_even")
            elif order_type == mt5.ORDER_TYPE_SELL and (current_sl is None or new_sl < current_sl):
                return self._modify_stop(position, new_sl, current_tp, "break_even")
        
        # --- Stage 3: Profit >= 2x trail distance → trail by trail_distance ---
        if profit_points >= trail_points * 2:
            if order_type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (trail_points * point)
            else:
                new_sl = current_price + (trail_points * point)
            
            new_sl = round(new_sl, digits)
            
            # Only modify if new SL is better than current
            if order_type == mt5.ORDER_TYPE_BUY and (current_sl is None or new_sl > current_sl):
                return self._modify_stop(position, new_sl, current_tp, "trailing")
            elif order_type == mt5.ORDER_TYPE_SELL and (current_sl is None or new_sl < current_sl):
                return self._modify_stop(position, new_sl, current_tp, "trailing")
        
        return None
    
    def _get_trail_points(self, symbol: str) -> float:
        """
        Get trailing stop distance in points.
        
        For US30/NAS100 indices: 50 points (as in ATS_US30_NAS)
        For XAUUSD: 100 points
        For Forex pairs: 20 points
        """
        if "XAU" in symbol or "GOLD" in symbol:
            return 100.0
        elif "US30" in symbol or "DJI" in symbol:
            return 50.0
        elif "NAS" in symbol or "US100" in symbol or "NDX" in symbol:
            return 50.0
        elif "JPY" in symbol:
            return 20.0
        else:
            return 50.0  # Default
    
    def _modify_stop(self, position, new_sl: float, new_tp: float, action_type: str) -> dict:
        """
        Modify SL/TP on an open position via MT5.
        """
        result = mt5.order_modify(
            position.ticket,
            price=position.price_open,
            sl=new_sl,
            tp=new_tp,
            expiration=position.expiration,
            stop_limit=position.price_open,
        )
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ {action_type.upper()}: Position {position.ticket} | "
                       f"New SL: {new_sl} | Pts: {(abs(new_sl - position.price_open) / position.price_open):.0f}")
            return {
                "ticket": position.ticket,
                "symbol": position.symbol,
                "action": action_type,
                "old_sl": position.sl,
                "new_sl": new_sl,
                "success": True,
            }
        else:
            logger.warning(f"⚠️ {action_type.upper()} FAILED: Position {position.ticket} | "
                          f"Retcode: {result.retcode if result else 'None'}")
            return {
                "ticket": position.ticket,
                "symbol": position.symbol,
                "action": action_type,
                "success": False,
                "retcode": result.retcode if result else -1,
            }
    
    def update_all_trailing_stops(self):
        """
        Public method: called from execution_agent.py to manage all trailing stops.
        Returns summary of actions taken.
        """
        actions = self.manage_trailing_stops()
        
        if actions:
            by_type = {}
            for a in actions:
                t = a.get("action", "unknown")
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(a)
            
            summary = {k: len(v) for k, v in by_type.items()}
            logger.info(f"Trailing stop actions: {summary}")
            return actions
        
        return []
