import pandas as pd
from pathlib import Path
from datetime import datetime, date
from core.logging_utils import get_logger

logger = get_logger(__name__)


class FlowDailyTracker:
    """Tracks FLOW_EXP trades per day and enforces daily limits."""
    
    def __init__(self, max_daily_trades=6):
        self.tracker_path = Path("data/live/flow_daily_count.csv")
        self.max_daily_trades = max_daily_trades
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create the tracker file with headers if it doesn't exist."""
        if not self.tracker_path.parent.exists():
            self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.tracker_path.exists():
            pd.DataFrame(columns=['date', 'signal_time', 'side', 'system']).to_csv(
                self.tracker_path, index=False
            )
    
    def get_today_count(self):
        """Get the number of FLOW trades executed today."""
        today = date.today()
        try:
            if not self.tracker_path.exists():
                return 0
            df = pd.read_csv(self.tracker_path, parse_dates=['signal_time'])
            if df.empty:
                return 0
            # Filter for today's date
            df['date'] = df['signal_time'].dt.date
            today_trades = df[df['date'] == today]
            return len(today_trades)
        except Exception as e:
            logger.error(f"Error counting today's FLOW trades: {e}")
            return 0
    
    def record_trade(self, signal_time, side, system):
        """Record a FLOW trade execution."""
        try:
            record = {
                'date': signal_time.date(),
                'signal_time': signal_time,
                'side': side,
                'system': system
            }
            if not self.tracker_path.exists():
                self._ensure_file_exists()
            df = pd.read_csv(self.tracker_path, parse_dates=['signal_time'])
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv(self.tracker_path, index=False)
            logger.info(f"Recorded FLOW trade for {signal_time}. Daily count: {self.get_today_count()}")
        except Exception as e:
            logger.error(f"Error recording FLOW trade: {e}")
    
    def is_limit_reached(self):
        """Check if the daily FLOW trade limit has been reached."""
        return self.get_today_count() >= self.max_daily_trades
    
    def get_remaining_trades(self):
        """Get the number of remaining FLOW trades allowed today."""
        return max(0, self.max_daily_trades - self.get_today_count())
    
    def get_daily_summary(self):
        """Get a summary of today's FLOW trading activity."""
        today = date.today()
        try:
            if not self.tracker_path.exists():
                return {'count': 0, 'buys': 0, 'sells': 0, 'remaining': self.max_daily_trades}
            df = pd.read_csv(self.tracker_path, parse_dates=['signal_time'])
            if df.empty:
                return {'count': 0, 'buys': 0, 'sells': 0, 'remaining': self.max_daily_trades}
            
            df['date'] = df['signal_time'].dt.date
            today_df = df[df['date'] == today]
            
            return {
                'count': len(today_df),
                'buys': len(today_df[today_df['side'] == 'BUY']),
                'sells': len(today_df[today_df['side'] == 'SELL']),
                'remaining': self.get_remaining_trades()
            }
        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            return {'count': 0, 'buys': 0, 'sells': 0, 'remaining': self.max_daily_trades}


# Singleton instance for consistent tracking across the application
_tracker_instance = None


def get_flow_tracker():
    """Get the singleton FlowDailyTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = FlowDailyTracker()
    return _tracker_instance