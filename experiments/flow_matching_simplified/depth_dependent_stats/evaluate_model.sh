#!/bin/bash
# Helper script to evaluate trained model with depth-dependent statistics

set -e

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Activate virtual environment (if exists)
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Make sure to activate it manually."
fi
DEPTH_STATS_DIR="$SCRIPT_DIR"

# Default values
CHECKPOINT=""
NUM_SAMPLES=1000
GENERATED_PROFILES=""
TARGET_STATS="$DEPTH_STATS_DIR/real_depth_stats.pkl"
TARGET_CORR="$DEPTH_STATS_DIR/real_correlations.pkl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/flow_matching_simplified/depth_dependent"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --generated-profiles)
            GENERATED_PROFILES="$2"
            shift 2
            ;;
        --target-stats)
            TARGET_STATS="$2"
            shift 2
            ;;
        --target-corr)
            TARGET_CORR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--checkpoint PATH] [--num-samples N] [--generated-profiles PATH] [--target-stats PATH] [--target-corr PATH] [--output-dir PATH]"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Export PYTHONPATH so Python can find the experiments module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Step 1: Generate samples if not provided
if [ -z "$GENERATED_PROFILES" ]; then
    echo "=========================================="
    echo "Step 1: Generating samples"
    echo "=========================================="
    
    if [ -z "$CHECKPOINT" ]; then
        # Try to find latest checkpoint in multiple possible locations
        CHECKPOINT_PATHS=(
            "experiments/flow_matching_simplified/outputs/flow_matching_simplified/checkpoints/latest.pt"
            "experiments/flow_matching_simplified/outputs/flow_matching_simplified/checkpoints/best.pt"
            "outputs/flow_matching_simplified/checkpoints/latest.pt"
            "outputs/flow_matching_simplified/checkpoints/best.pt"
            "$(find experiments/flow_matching_simplified/outputs -name 'latest.pt' -type f 2>/dev/null | head -1)"
            "$(find experiments/flow_matching_simplified/outputs -name 'best.pt' -type f 2>/dev/null | head -1)"
            "$(find outputs -name 'latest.pt' -type f 2>/dev/null | head -1)"
            "$(find outputs -name 'best.pt' -type f 2>/dev/null | head -1)"
        )
        
        CHECKPOINT=""
        for path in "${CHECKPOINT_PATHS[@]}"; do
            if [ -n "$path" ] && [ -f "$path" ]; then
                CHECKPOINT="$path"
                echo "Found checkpoint: $CHECKPOINT"
                break
            fi
        done
        
        if [ -z "$CHECKPOINT" ]; then
            echo "Error: No checkpoint found in common locations."
            echo "Please specify --checkpoint PATH"
            echo ""
            echo "You can also skip sample generation by providing existing samples:"
            echo "  $0 --generated-profiles <path-to-samples.npy>"
            exit 1
        fi
    fi
    
    GENERATED_PROFILES="experiments/flow_matching_simplified/outputs/flow_matching_simplified/samples/generated_samples_eval.npy"
    
    python experiments/flow_matching_simplified/sample.py \
        --checkpoint "$CHECKPOINT" \
        --num_samples "$NUM_SAMPLES" \
        --output_path "$GENERATED_PROFILES"
    
    echo ""
fi

# Step 2: Evaluate depth-dependent statistics
echo "=========================================="
echo "Step 2: Evaluating depth-dependent statistics"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

EVAL_ARGS=(
    --generated-profiles "$GENERATED_PROFILES"
    --target-stats "$TARGET_STATS"
    --target-corr "$TARGET_CORR"
    --output-dir "$OUTPUT_DIR"
)

python experiments/flow_matching_simplified/depth_dependent_stats/evaluate_depth_stats.py "${EVAL_ARGS[@]}"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo "Generated profiles: $GENERATED_PROFILES"
echo "Target statistics: $TARGET_STATS"
echo "Target correlations: $TARGET_CORR"
echo "Output directory: $OUTPUT_DIR"

