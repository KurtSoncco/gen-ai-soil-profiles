#!/usr/bin/env python3
"""
VAE Test Runner
Run all tests for the VAE experiments.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file):
    """Run a single test file."""
    print(f"🧪 Running {test_file}...")
    try:
        # Change to the VAE directory and run the test
        vae_dir = Path(__file__).parent
        result = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True, cwd=vae_dir
        )
        if result.returncode == 0:
            print(f"✅ {test_file} passed")
            return True
        else:
            print(f"❌ {test_file} failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False


def main():
    """Run all tests."""
    script_dir = Path(__file__).parent
    tests_dir = script_dir / "tests"

    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return

    test_files = ["tests/smoke_test.py", "tests/test_pipeline.py"]

    print("🧪 Running VAE Tests")
    print("=" * 40)

    passed = 0
    total = len(test_files)

    for test_file in test_files:
        if run_test(test_file):
            passed += 1
        print()

    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
