#!/bin/bash
# Helper script to train model with depth-dependent statistics losses

set -e

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEPTH_STATS_DIR="$SCRIPT_DIR"

# Activate virtual environment (if exists)
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Make sure to activate it manually."
fi

# Default values
TARGET_STATS="$DEPTH_STATS_DIR/real_depth_stats.pkl"
TARGET_CORR="$DEPTH_STATS_DIR/real_correlations.pkl"
DEPTH_STATS_WEIGHT=""
VERTICAL_CORR_WEIGHT=""
RESUME_CHECKPOINT=""
NUM_EPOCHS=""
DEVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target-stats)
            TARGET_STATS="$2"
            shift 2
            ;;
        --target-corr)
            TARGET_CORR="$2"
            shift 2
            ;;
        --depth-stats-weight)
            DEPTH_STATS_WEIGHT="$2"
            shift 2
            ;;
        --vertical-corr-weight)
            VERTICAL_CORR_WEIGHT="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train flow matching model with depth-dependent statistics losses"
            echo ""
            echo "Options:"
            echo "  --target-stats PATH         Path to depth statistics .pkl file"
            echo "                             (default: $DEPTH_STATS_DIR/real_depth_stats.pkl)"
            echo "  --target-corr PATH         Path to correlations .pkl file"
            echo "                             (default: $DEPTH_STATS_DIR/real_correlations.pkl)"
            echo "  --depth-stats-weight FLOAT Weight for depth statistics loss (overrides config)"
            echo "  --vertical-corr-weight FLOAT Weight for vertical correlation loss (overrides config)"
            echo "  --num-epochs INT           Override number of epochs (default: 1000)"
            echo "  --device DEVICE            Override device (cuda or cpu)"
            echo "  --resume PATH              Resume training from checkpoint"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "The 1000-epoch flagship run requires a CUDA GPU (Lambda or Savio)."
            echo "Set FORCE_CPU_TRAIN=1 to override that check."
            echo ""
            echo "Examples:"
            echo "  # Train with default settings"
            echo "  $0"
            echo ""
            echo "  # Train with custom loss weights"
            echo "  $0 --depth-stats-weight 0.01 --vertical-corr-weight 0.03"
            echo ""
            echo "  # Train with custom statistics files"
            echo "  $0 --target-stats /path/to/stats.pkl --target-corr /path/to/corr.pkl"
            echo ""
            echo "  # Resume training from checkpoint"
            echo "  $0 --resume outputs/flow_matching_simplified/checkpoints/best.pt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Export PYTHONPATH so Python can find the experiments module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Disable wandb when no API key is configured
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    echo "wandb disabled (WANDB_API_KEY not set)"
fi

# Flagship 1000-epoch run expects a GPU. Override with FORCE_CPU_TRAIN=1.
if [ "${FORCE_CPU_TRAIN:-}" != "1" ] && [ "${DEVICE:-}" != "cpu" ]; then
    if ! python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    then
        echo "Error: no CUDA GPU detected."
        echo "The flagship 1000-epoch depth-aware run should be launched on Lambda or Savio."
        echo "Start a GPU worker (cursor worker start) or set FORCE_CPU_TRAIN=1 to override."
        exit 1
    fi
fi

# Check if target statistics files exist
if [ ! -f "$TARGET_STATS" ]; then
    echo "Error: Target statistics file not found: $TARGET_STATS"
    echo ""
    echo "Please precompute statistics first:"
    echo "  python experiments/flow_matching_simplified/depth_dependent_stats/precompute_statistics.py"
    echo ""
    echo "Or specify a different path with --target-stats"
    exit 1
fi

if [ ! -f "$TARGET_CORR" ]; then
    echo "Warning: Target correlations file not found: $TARGET_CORR"
    echo "Training will continue without vertical correlation loss"
fi

# Build training command
TRAIN_ARGS=(
    --target-stats "$TARGET_STATS"
)

# Add target correlations if file exists
if [ -f "$TARGET_CORR" ]; then
    TRAIN_ARGS+=(--target-corr "$TARGET_CORR")
fi

# Add loss weights if specified
if [ -n "$DEPTH_STATS_WEIGHT" ]; then
    TRAIN_ARGS+=(--depth-stats-weight "$DEPTH_STATS_WEIGHT")
fi

if [ -n "$VERTICAL_CORR_WEIGHT" ]; then
    TRAIN_ARGS+=(--vertical-corr-weight "$VERTICAL_CORR_WEIGHT")
fi

if [ -n "$NUM_EPOCHS" ]; then
    TRAIN_ARGS+=(--num-epochs "$NUM_EPOCHS")
fi

if [ -n "$DEVICE" ]; then
    TRAIN_ARGS+=(--device "$DEVICE")
fi

# Display configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Target statistics: $TARGET_STATS"
echo "Target correlations: $TARGET_CORR"
if [ -n "$DEPTH_STATS_WEIGHT" ]; then
    echo "Depth stats weight: $DEPTH_STATS_WEIGHT"
fi
if [ -n "$VERTICAL_CORR_WEIGHT" ]; then
    echo "Vertical corr weight: $VERTICAL_CORR_WEIGHT"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
fi
echo "=========================================="
echo ""

# Note: The training script doesn't currently support --resume, but we can add it later
# For now, just warn if user tries to use it
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Warning: --resume is not yet implemented in the training script."
    echo "You may need to modify train_with_depth_losses.py to support checkpoint resuming."
    echo ""
fi

# Run training
echo "Starting training..."
echo ""

python experiments/flow_matching_simplified/depth_dependent_stats/train_with_depth_losses.py "${TRAIN_ARGS[@]}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: outputs/flow_matching_simplified/checkpoints/"
echo "Results saved to: outputs/flow_matching_simplified/results/"
echo ""
echo "To evaluate the trained model, run:"
echo "  ./experiments/flow_matching_simplified/depth_dependent_stats/evaluate_model.sh"

