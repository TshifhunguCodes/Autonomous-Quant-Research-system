"""
Session Filter — News Blackout Windows & Weekend Guard
Adapted from ATS_US30_NAS into AQRS

Blocks trading during:
1. Weekends (Saturday/Sunday)
2. High-impact news events (FOMC, NFP, CPI, etc.)
3. Illiquid session times
"""
from datetime import datetime, time, timedelta
import logging
import pytz

logger = logging.getLogger(__name__)


class SessionFilter:
    """
    Filter that blocks trading during specific sessions and news events.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.utc = pytz.UTC
        
        # Weekend guard: No trading on Saturday or Sunday
        self.block_weekends = True
        
        # Session hours (UTC): when trading is allowed
        self.session_start_hour = 0    # 00:00 UTC (Asian open)
        self.session_end_hour = 21      # 21:00 UTC (US close)
        
        # News blackout windows (in UTC, using datetime ranges)
        # Key: event name, Value: list of (month, day, hour, duration_hours)
        # These are typical US session high-impact events
        self.news_events = [
            # FOMC (Federal Reserve) — typically 14:00-15:00 UTC, 8 times/year
            {"name": "FOMC", "duration_before": 2, "duration_after": 2},
            
            # NFP (Non-Farm Payrolls) — first Friday of month, 12:30-13:30 UTC
            {"name": "NFP", "duration_before": 2, "duration_after": 2},
            
            # CPI (Consumer Price Index) — varies, typically 12:30-13:30 UTC
            {"name": "CPI", "duration_before": 1, "duration_after": 1},
            
            # FOMC Minutes — 18:00 UTC
            {"name": "FOMC_MINUTES", "duration_before": 1, "duration_after": 1},
        ]
        
        # Fixed blackout windows (every day)
        # Optional: block during rollover (22:00-00:00 UTC for forex)
        self.rollover_blackout = False  # Set to True to block during rollover
    
    def is_weekend(self, dt: datetime = None) -> bool:
        """Check if datetime falls on a weekend."""
        if dt is None:
            dt = datetime.now(self.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.utc)
        return dt.weekday() >= 5  # 5=Saturday, 6=Sunday
    
    def is_outside_session(self, dt: datetime = None) -> bool:
        """Check if datetime is outside trading session hours."""
        if dt is None:
            dt = datetime.now(self.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.utc)
        
        hour = dt.hour
        return hour < self.session_start_hour or hour >= self.session_end_hour
    
    def is_news_blackout(self, dt: datetime = None, event_name: str = None) -> dict:
        """
        Check if datetime falls within a news blackout window.
        
        This is a template. Real implementation would need to:
        - Fetch economic calendar from API (e.g., ForexFactory, Investing.com)
        - Parse exact event timestamps
        
        For now, we return False (no blackout) and log a warning.
        In production, integrate with an economic calendar API.
        
        Args:
            dt: Datetime to check (default: now)
            event_name: Optional specific event to check
        
        Returns:
            dict with: in_blackout (bool), event (str), reason (str)
        """
        if dt is None:
            dt = datetime.now(self.utc)
        
        # TODO: Integrate with real economic calendar API
        # For production, use:
        # - ForexFactory API (https://www.forexfactory.com/calendar)
        # - Investing.com API
        # - Alpha Vantage economic indicators
        
        return {
            "in_blackout": False,
            "event": "",
            "reason": "no_calendar_integration_signal_allowed",
        }
    
    def can_trade(self, dt: datetime = None) -> dict:
        """
        Full session check for a given datetime.
        
        Args:
            dt: Datetime to check (default: now UTC)
        
        Returns:
            dict with: can_trade (bool), reason (str)
        """
        if dt is None:
            dt = datetime.now(self.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.utc)
        
        # Check 1: Weekend
        if self.block_weekends and self.is_weekend(dt):
            return {"can_trade": False, "reason": "weekend_guard"}
        
        # Check 2: Session hours
        if self.is_outside_session(dt):
            return {"can_trade": False, "reason": f"outside_session_hours_{self.session_start_hour}-{self.session_end_hour}UTC"}
        
        # Check 3: News blackout
        news_check = self.is_news_blackout(dt)
        if news_check["in_blackout"]:
            return {"can_trade": False, "reason": news_check["reason"]}
        
        # Check 4: Rollover
        if self.rollover_blackout and 22 <= dt.hour < 24:
            return {"can_trade": False, "reason": "rollover_blackout"}
        
        return {"can_trade": True, "reason": "session_allowed"}
    
    def filter_signal(self, signal_row: dict) -> dict:
        """
        Check if a signal should be allowed based on current session.
        
        Args:
            signal_row: Dict with signal metadata (may contain 'current_time')
        
        Returns:
            dict with: allowed (bool), reason (str)
        """
        signal_time = signal_row.get("current_time")
        if signal_time is not None:
            if isinstance(signal_time, str):
                signal_time = datetime.fromisoformat(signal_time)
        
        result = self.can_trade(signal_time)
        
        return {
            "allowed": result["can_trade"],
            "reason": result["reason"],
            "filter": "session_filter",
        }