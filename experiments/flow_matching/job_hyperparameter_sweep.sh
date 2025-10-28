#!/bin/bash
#SBATCH --job-name=ffm_hp_sweep
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case):
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=8
#
# Number and type of GPUs
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal

# Wall clock limit (set to 2 hours for multiple experiments):
#SBATCH --time=02:00:00

## Load required modules
module load anaconda3 cuda cudnn pytorch gcc/13.2.0 openblas/0.3.24

## Set working directory to project root
cd /global/home/users/kurtwal98/gen-ai-soil-profiles

## Activate virtual environment
source .venv/bin/activate

## Navigate to experiment directory
cd experiments/flow_matching

# Configuration
NUM_STEPS=2500
BASE_PARAMS="--num_steps $NUM_STEPS"

# Function to run training with given parameters
run_experiment() {
    local name=$1
    shift
    local params="$@"
    
    echo "========================================="
    echo "Running experiment: $name"
    echo "Parameters: $params"
    echo "========================================="
    
    python train.py $BASE_PARAMS --wandb_name "$name" $params
    
    echo "Completed experiment: $name"
    echo ""
}

# Base case with all parameters at 1.0 (for PCFM) and 0.01 for TVD
run_experiment "base_all_1.0" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 1.0 --tvd_weight 0.01"

# pcfm_guidance_strength variations (keep monotonic=1.0, tvd_weight=0.01)
run_experiment "guidance_0.2" "--use_pcfm true --pcfm_guidance_strength 0.2 --pcfm_monotonic_weight 1.0 --tvd_weight 0.01"
run_experiment "guidance_0.1" "--use_pcfm true --pcfm_guidance_strength 0.1 --pcfm_monotonic_weight 1.0 --tvd_weight 0.01"
run_experiment "guidance_0.05" "--use_pcfm true --pcfm_guidance_strength 0.05 --pcfm_monotonic_weight 1.0 --tvd_weight 0.01"
run_experiment "guidance_0.0" "--use_pcfm true --pcfm_guidance_strength 0.0 --pcfm_monotonic_weight 1.0 --tvd_weight 0.01"

# pcfm_monotonic_weight variations (keep guidance=1.0, tvd_weight=0.01)
run_experiment "monotonic_0.2" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 0.2 --tvd_weight 0.01"
run_experiment "monotonic_0.1" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 0.1 --tvd_weight 0.01"
run_experiment "monotonic_0.05" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 0.05 --tvd_weight 0.01"
run_experiment "monotonic_0.0" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 0.0 --tvd_weight 0.01"

# tvd_weight variations (keep guidance=1.0, monotonic=1.0, no PCFM for TVD-only tests)
run_experiment "tvd_0.2" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 1.0 --tvd_weight 0.2"
run_experiment "tvd_0.1" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 1.0 --tvd_weight 0.1"
run_experiment "tvd_0.05" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 1.0 --tvd_weight 0.05"
run_experiment "tvd_0.0" "--use_pcfm true --pcfm_guidance_strength 1.0 --pcfm_monotonic_weight 1.0 --tvd_weight 0.0"

echo "========================================="
echo "All experiments completed!"
echo "========================================="

