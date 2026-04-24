import MetaTrader5 as mt5
import sys
import os
from pathlib import Path

def check_environment():
    print("==================================================")
    print("        AQRS V2 ENVIRONMENT VERIFICATION          ")
    print("==================================================")
    
    # 1. Python Check
    print(f"Python Version: {sys.version}")
    
    # 2. Package Check
    print("\n--- Package Verification ---")
    packages = ["pandas", "MetaTrader5", "streamlit", "fastapi", "uvicorn", "requests"]
    for pkg in packages:
        try:
            __import__(pkg)
            status = "✅ OK"
        except ImportError:
            status = "❌ MISSING"
        print(f"{pkg:<20} {status}")

    # 2. Path Check
    root = Path(__file__).parent.parent
    required_dirs = ["data/raw", "data/research", "data/backtest", "logs", "config"]
    print("\n--- Directory Audit ---")
    for d in required_dirs:
        p = root / d
        status = "✅ OK" if p.exists() else "❌ MISSING"
        print(f"{d:<20} {status}")

    # 3. MT5 Check
    print("\n--- MT5 Connection Audit ---")
    if not mt5.initialize():
        print("MT5 Status: ❌ FAILED. Ensure MT5 is installed and active.")
    else:
        acc = mt5.account_info()
        if acc:
            mode = "DEMO" if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "LIVE"
            print(f"MT5 Status: ✅ CONNECTED")
            print(f"Account:    {acc.login} ({acc.company})")
            print(f"Trade Mode: {mode}")
        mt5.shutdown()
    print("==================================================\n")

if __name__ == "__main__":
    check_environment()