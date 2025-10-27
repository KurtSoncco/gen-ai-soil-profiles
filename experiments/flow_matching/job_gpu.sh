#!/bin/bash
#SBATCH --job-name=test 
#SBATCH --account=fc_tfsurrogate 
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Four times the number of GPUs for A40 in savio3_gpu
#SBATCH --cpus-per-task=4
#
#Number of GPUs
#SBATCH --gres=gpu:A40:1

#SBATCH --qos=savio_normal

# Wall clock limit:
#SBATCH --time=00:00:30
## Command(s) to run (example):
./a.out