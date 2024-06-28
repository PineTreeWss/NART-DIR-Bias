#!/bin/bash
#SBATCH --job-name          BC-T-Inference
#SBATCH --gres              gpu:1
#SBATCH --mem               64G

python Inference_Valid_bi.py
