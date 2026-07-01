#!/bin/bash
# Helper script to compare multiple model outputs: Vs vs Depth visualization

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

# Default values
OUTPUT="vs_vs_depth_comparison.png"
MODELS=()
N_PROFILES=30
NO_REAL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [ $# -lt 3 ]; then
                echo "Error: --model requires two arguments: PROFILES_PATH and MODEL_NAME"
                exit 1
            fi
            MODELS+=("$2" "$3")
            shift 3
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --n-profiles)
            N_PROFILES="$2"
            shift 2
            ;;
        --no-real)
            NO_REAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Compare multiple model outputs: Vs vs Depth visualization"
            echo ""
            echo "Options:"
            echo "  --model PROFILES NAME    Generated profiles .npy file and model name"
            echo "                           (can be used multiple times to add more models)"
            echo "  --output PATH            Output path for comparison plot"
            echo "                           (default: vs_vs_depth_comparison.png)"
            echo "  --n-profiles N           Number of profiles to plot per panel (default: 30)"
            echo "  --no-real                Don't include real data panel (only show models)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Compare two models"
            echo "  $0 \\"
            echo "      --model outputs/model1/samples.npy \"Baseline Model\" \\"
            echo "      --model outputs/model2/samples.npy \"Improved Model\" \\"
            echo "      --output comparison.png"
            echo ""
            echo "  # Compare multiple models"
            echo "  $0 \\"
            echo "      --model model1.npy \"Model 1\" \\"
            echo "      --model model2.npy \"Model 2\" \\"
            echo "      --model model3.npy \"Model 3\" \\"
            echo "      --output comparison.png"
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

# Validate inputs
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: At least one --model argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check that model files exist
for ((i=0; i<${#MODELS[@]}; i+=2)); do
    profiles_path="${MODELS[$i]}"
    if [ ! -f "$profiles_path" ]; then
        echo "Error: Model profiles file not found: $profiles_path"
        exit 1
    fi
done

# Build comparison command
COMPARE_ARGS=(
    --output "$OUTPUT"
    --n-profiles "$N_PROFILES"
)

if [ "$NO_REAL" = true ]; then
    COMPARE_ARGS+=(--no-real)
fi

# Add all models
for ((i=0; i<${#MODELS[@]}; i+=2)); do
    COMPARE_ARGS+=(--model "${MODELS[$i]}" "${MODELS[$i+1]}")
done

# Display configuration
echo "=========================================="
echo "Vs vs Depth Comparison Configuration"
echo "=========================================="
if [ "$NO_REAL" = false ]; then
    echo "Real data: Included (from test set)"
else
    echo "Real data: Excluded"
fi
echo ""
echo "Models to compare:"
for ((i=0; i<${#MODELS[@]}; i+=2)); do
    echo "  - ${MODELS[$i+1]}: ${MODELS[$i]}"
done
echo ""
echo "Profiles per panel: $N_PROFILES"
echo "Output: $OUTPUT"
echo "=========================================="
echo ""

# Run comparison
python experiments/flow_matching_simplified/depth_dependent_stats/compare_models.py "${COMPARE_ARGS[@]}"

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
echo "Comparison plot saved to: $OUTPUT"

