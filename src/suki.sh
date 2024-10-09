#!/bin/bash

#SBATCH --mail-user=username@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=slurm_output/out%j.out
#SBATCH --error=slurm_output/err%j.err

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -p academic
#SBATCH -t 23:00:00

module load miniconda3

source activate
conda activate lab3

module load cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

python3 custom_mlp.py