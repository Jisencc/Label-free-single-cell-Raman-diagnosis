#!/bin/bash
#SBATCH --job-name=test-updcell
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=167:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=31

#python -c "from clustering import stage1_clustering_global_and_per_celltype; stage1_clustering_global_and_per_celltype('raman.csv','results_stage1_cluster')"
#python raman_stage1_debug.py --raman_csv raman.csv --out_dir results_debug
python raman_stage1_binary_publication_final.py

#python -c "from clustering import train_stage1_mil; train_stage1_mil('raman.csv','results_stage1_mil')"

