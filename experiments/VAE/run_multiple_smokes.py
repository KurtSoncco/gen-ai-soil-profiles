#!/usr/bin/env python3
"""
Run the full experiment multiple times with a short delay to observe performance.
Uses src/vae_implementation.py main entry.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_once(idx: int) -> int:
    print(f"\n=== Run {idx + 1} ===")
    vae_dir = Path(__file__).parent
    # Run full experiment
    proc = subprocess.run([sys.executable, "run_experiment.py"], cwd=vae_dir)
    return proc.returncode


def main() -> int:
    runs = 3
    delay_sec = 3
    passed = 0
    for i in range(runs):
        code = run_once(i)
        if code == 0:
            passed += 1
        if i < runs - 1:
            time.sleep(delay_sec)
    print(f"\nCompleted {runs} runs. Passed: {passed}/{runs}")
    return 0 if passed == runs else 1


if __name__ == "__main__":
    sys.exit(main())
