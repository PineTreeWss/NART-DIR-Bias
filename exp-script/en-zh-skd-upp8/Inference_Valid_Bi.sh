#!/bin/bash
#SBATCH --job-name          DT-Hybrid
#SBATCH --time              48:00:00
#SBATCH --cpus-per-task     16
#SBATCH --gres              gpu:1
#SBATCH --mem               64G
#SBATCH --output            DA-T-Ro-En.%j.out
#SBATCH --partition         a100_batch

python Inference_Valid_bi.py