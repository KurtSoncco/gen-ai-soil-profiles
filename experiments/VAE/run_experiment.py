#!/usr/bin/env python3
"""
VAE Experiment Runner
Simple script to run the main VAE experiment from the organized directory structure.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the main VAE experiment."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up to project root
    
    print("🚀 Starting VAE experiment...")
    print(f"📁 Working directory: {project_root}")
    print("=" * 50)
    
    try:
        # Run the experiment using the module approach that works
        result = subprocess.run([
            sys.executable, "-m", "experiments.VAE.src.vae_implementation"
        ], cwd=project_root, check=True)
        
        print("✅ VAE experiment completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running experiment: {e}")
        print("Make sure you're in the project root and all dependencies are installed.")
        return e.returncode
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    main()
