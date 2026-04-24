import subprocess
import os
import sys
from pathlib import Path

def run_smoke_test():
    root = Path(__file__).parent.parent
    os.chdir(root)
    
    print("🚀 Starting AQRS Smoke Test (Research Logic Validation)...")
    
    # Run research mode using existing artifacts to verify transform integrity
    cmd = [sys.executable, "main.py", "--mode", "research", "--reuse-artifacts"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Smoke Test PASSED: Strategy pipeline executed without errors.")
        else:
            print("❌ Smoke Test FAILED")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("❌ Smoke Test FAILED: Execution timed out.")
    except Exception as e:
        print(f"❌ Smoke Test FAILED: {str(e)}")

if __name__ == "__main__":
    run_smoke_test()