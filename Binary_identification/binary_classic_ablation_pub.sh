#!/bin/bash
#SBATCH --job-name=test-updcell
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=167:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=31


python raman_stage1_binary_fpch_ablation_publication_final_CV3.py



