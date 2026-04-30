import pandas as pd
from pathlib import Path
from core.config import load_config
from config.v3_config import V3Config
from strategy import execution_agent

def main():
    print("🔍 Loading AQRS V3 configuration...")
    base_config = load_config()
    config = V3Config.load_from(base_config)
    
    path = Path("data/features/trade_setups.csv")
    print(f"📂 Scanning historical signals in {path}...")
    
    if not path.exists():
        print("❌ No historical signals found. Sending a MOCK signal for connection test...")
        execution_agent._send_mobile_alert(
            config, "ALPHA", "buy", 2450.50, 2440.00, 2470.00, "ELITE", "XAUUSD", "ALIGNED_TREND"
        )
        return

    df = pd.read_csv(path)
    # Find the most recent signal that was actually a trade (buy/sell)
    signals = df[df['confirmed_signal'].isin(['buy', 'sell'])]
    
    if signals.empty:
        print("⚠️ No valid buy/sell signals found in history. Sending a MOCK signal instead...")
        execution_agent._send_mobile_alert(
            config, "ALPHA", "buy", 2450.50, 2440.00, 2470.00, "ELITE", "XAUUSD", "ALIGNED_TREND"
        )
    else:
        last = signals.iloc[-1]
        print(f"✅ Found historical signal from {last['time']}. Pushing to Telegram...")
        execution_agent._send_mobile_alert(
            config,
            last.get("system", "ALPHA"),
            last["confirmed_signal"],
            last["entry_price"],
            last["stop_loss"],
            last["take_profit"],
            last["quality"],
            config.market.symbol,
            last.get("market_regime", "UNKNOWN")
        )

if __name__ == "__main__":
    main()