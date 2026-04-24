import subprocess
import sys
import os
import time
from pathlib import Path

def launch():
    root = Path(__file__).parent.parent
    os.chdir(root)
    
    print("🔥 Launching AQRS Real-Time Dashboard...")
    
    # 1. Start FastAPI Backend
    api_cmd = [sys.executable, "-m", "uvicorn", "strategy.backend_api:app", "--port", "8001"]
    api_proc = subprocess.Popen(api_cmd)
    
    # Give API time to bind
    time.sleep(2)
    
    # 2. Start Streamlit Frontend
    st_cmd = [sys.executable, "-m", "streamlit", "run", "strategy/streamlit_app.py"]
    st_proc = subprocess.Popen(st_cmd)
    
    try:
        api_proc.wait()
        st_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        api_proc.terminate()
        st_proc.terminate()

if __name__ == "__main__":
    launch()