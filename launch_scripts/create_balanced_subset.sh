#!/bin/sh
#SBATCH --job-name=xg_boost
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH --mem=128G
#SBATCH -C=gpu
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=balanced.log
#SBATCH --error=balanced.err
module load conda
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn python create_balanced_subset.py
