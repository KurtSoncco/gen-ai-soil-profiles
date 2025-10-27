#!/bin/bash
#SBATCH --job-name=ffm_training
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case):
#SBATCH --ntasks=1
#
# Processors per task:
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=8
#
# Number and type of GPUs
#SBATCH --gres=gpu:a40:1

#SBATCH --qos=a40_gpu3_normal

# Wall clock limit (set to 40 minutes for training):
#SBATCH --time=00:40:00

## Load required modules
module load anaconda3 cuda cudnn pytorch gcc/13.2.0 openblas/0.3.24

## Set working directory to project root
cd /global/home/users/kurtwal98/gen-ai-soil-profiles

## Activate virtual environment
source .venv/bin/activate

## Navigate to experiment directory
cd experiments/flow_matching

## Command to run training:
python train.py