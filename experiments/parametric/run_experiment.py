#!/usr/bin/env python3
"""
Simple runner script for the parametric profile modeling experiment.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the parametric experiment."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    print("🚀 Starting Parametric Profile Modeling Experiment...")
    print(f"📁 Working directory: {project_root}")
    print("=" * 50)

    try:
        # Run the main experiment
        subprocess.run(
            [sys.executable, "run_parametric_experiment.py"], cwd=script_dir, check=True
        )

        print("✅ Parametric experiment completed successfully!")

        # Run evaluation
        print("\n🔍 Running evaluation...")
        subprocess.run(
            [sys.executable, "evaluate_parametric.py"], cwd=script_dir, check=True
        )

        print("✅ Evaluation completed successfully!")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running experiment: {e}")
        return e.returncode
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
